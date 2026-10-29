"""Replayable, verifiable decision receipts (novelty #4).

Runs the real absence-recovery flow on a fresh synthetic organisation (D1),
replays every recorded decision from its ledger receipt, and then attacks the
record. Writes results/receipt_report.json for the paper.
"""
import copy
import json
import os
import tempfile
from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models, schemas
from app.ai import audit_ledger, bigru, decision_receipt as dr, workload_engine
from app.routers import absence

REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "receipt_report.json")
_ATTACKS = []


# ------------------------------------------------------------------ fixtures / helpers
@pytest.fixture(scope="module")
def flow():
    from app.database import Base
    from app import seed as seed_module
    from app.ai import rules_engine
    path = os.path.join(tempfile.mkdtemp(), "receipts.db")
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    seed_module.seed(db)
    rules_engine.ensure_default_rules(db)
    bigru.train(workload_engine.build_sequences_for_training(db))   # exercise the risk gate
    # absent the employee with the most open work, so several decisions are made in one batch
    emps = db.query(models.Employee).all()
    load = {e.id: db.query(models.Issue).filter(models.Issue.assignee_id == e.id,
                                                models.Issue.status != "Done").count() for e in emps}
    absent = max(load, key=load.get)
    results = absence.mark_absent(schemas.AbsenceRequest(employee_id=absent), db)
    db.expire_all()   # force everything below to be read back from the database (JSON round trip)
    events = db.query(models.AuditEvent).filter(models.AuditEvent.event_type == "REALLOCATION_RECOMMENDED") \
        .order_by(models.AuditEvent.id.asc()).all()
    yield {"db": db, "results": results, "events": events, "absent": absent}
    db.close()


def _resign_from(db, event):
    """Attacker WITH the signing key: recompute hash + signature of this event and every later one."""
    later = db.query(models.AuditEvent).filter(models.AuditEvent.id >= event.id).order_by(models.AuditEvent.id).all()
    prev = event.previous_hash
    for e in later:
        e.previous_hash = prev
        e.event_hash = audit_ledger.compute_event_hash(e.payload_json, prev)
        e.signature = audit_ledger.sign_hash(e.event_hash)
        prev = e.event_hash
    db.commit()


def _edit_payload(db, event, mutate):
    p = copy.deepcopy(event.payload_json)
    mutate(p)
    event.payload_json = p
    db.commit()


def _restore(db, event, payload, prev, h, sig):
    event.payload_json = payload
    event.previous_hash, event.event_hash, event.signature = prev, h, sig
    db.commit()
    _resign_from(db, event)   # restores later events too (same key, same payloads -> same hashes)


def _attack(name, actor, expected, observed, detected):
    _ATTACKS.append({"attack": name, "attacker": actor, "expected": expected, "observed": observed,
                     "detected": detected})


# ------------------------------------------------------------------ honest replay
def test_every_decision_replays(flow):
    db, events = flow["db"], flow["events"]
    assert len(events) >= 2, "need several decisions to be meaningful"
    out = [dr.verify_decision(db, e.event_id) for e in events]
    statuses = [o["status"] for o in out]
    assert statuses == ["VERIFIED"] * len(events), [o.get("replay", {}).get("mismatches") for o in out]
    assert all(o["replay"]["code_fingerprint_matches"] for o in out)
    flow["verified"] = len(events)
    flow["checks"] = sum(o["replay"]["checks_run"] for o in out)


def test_receipt_contains_gate_and_inputs(flow):
    r = flow["events"][0].payload_json["data"]["receipt"]
    assert r["gates"]["risk"]["record"]["status"] in ("gated_promoted", "gated_demoted")
    assert r["inputs"] and all("recovery_risk_baseline" in i for i in r["inputs"])
    assert r["policy"]["rules_hash"] == dr.sha256_json(r["policy"]["rules"])


def test_concurrent_absence_adjustment_is_replayed(flow):
    ctxs = [e.payload_json["data"]["receipt"]["context"] for e in flow["events"]]
    flow["adjusted"] = sum(1 for c in ctxs if c.get("prior_picks_of_top", 0) > 0)
    # whether or not an adjustment happened, every decision verified above, including adjusted ones


# ------------------------------------------------------------------ manager decision binding
def test_approval_bound_to_recommendation(flow):
    db = flow["db"]
    ev = next(e for e in flow["events"] if e.payload_json["data"]["recommended_employee_id"])
    task = ev.task_id
    rec_id = ev.payload_json["data"]["recommended_employee_id"]
    other = next(c["employee_id"] for c in ev.payload_json["data"]["receipt"]["outputs"]
                 if c["employee_id"] != rec_id and not c["excluded_by_rule"])
    absence.approve_reassignment(task, other, decision="approved", db=db)   # declared approved, actually modified
    db.expire_all()
    a = audit_ledger.latest_event(db, task, ["RECOMMENDATION_APPROVED", "RECOMMENDATION_MODIFIED"])
    d = a.payload_json["data"]
    assert a.event_type == "RECOMMENDATION_MODIFIED"
    assert d["followed_recommendation"] is False and d["declared_action_consistent"] is False
    assert d["recommendation_event_hash"] == ev.event_hash
    assert d["receipt_hash"] == dr.receipt_hash(ev.payload_json["data"]["receipt"])
    assert audit_ledger.verify_full_chain(db)["status"] == "VALID"


# ------------------------------------------------------------------ attacks
def _snapshot(e):
    return copy.deepcopy(e.payload_json), e.previous_hash, e.event_hash, e.signature


def test_A1_flip_gate_verdict_without_key(flow):
    db, e = flow["db"], flow["events"][0]
    snap = _snapshot(e)
    _edit_payload(db, e, lambda p: p["data"]["receipt"]["gates"]["risk"]["record"].update(status="gated_promoted"))
    st = dr.verify_decision(db, e.event_id)["status"]
    _restore(db, e, *snap)
    _attack("Flip gate verdict in receipt", "no key", "detected (hash)", st, st == "CHAIN_BROKEN")
    assert st == "CHAIN_BROKEN"


def test_A2_flip_gate_verdict_with_key(flow):
    """Key holder claims the BiGRU was promoted (and fixes the record hash) to legitimise a model-driven score."""
    db, e = flow["db"], flow["events"][0]
    snap = _snapshot(e)

    def m(p):
        g = p["data"]["receipt"]["gates"]["risk"]
        g["record"]["status"] = "gated_demoted" if g["record"]["status"] == "gated_promoted" else "gated_promoted"
        g["record_hash"] = dr.sha256_json(g["record"])
    _edit_payload(db, e, m)
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    failed = {x["check"].split(":")[0] for x in v["replay"]["mismatches"]} if v.get("replay") else set()
    _attack("Flip gate verdict and re-sign", "key holder", "detected (replay: gate enforcement)",
            f"{v['status']} {sorted(failed)}", v["status"] == "REPLAY_MISMATCH")
    assert v["ledger"]["status"] == "VALID" and v["status"] == "REPLAY_MISMATCH"
    assert failed & {"gate_enforced", "gate_status_consistent", "segment_gate_consistent"}


def test_A3_serve_demoted_model_with_key(flow):
    """Key holder rewrites a candidate's served risk to the model value while the gate says demoted."""
    db, e = flow["db"], flow["events"][0]
    r = e.payload_json["data"]["receipt"]
    if r["gates"]["risk"]["record"]["status"] != "gated_demoted":
        pytest.skip("risk gate promoted on this seed")
    snap = _snapshot(e)

    def m(p):
        inp = next(i for i in p["data"]["receipt"]["inputs"] if i["recovery_risk_model"] is not None)
        inp["recovery_risk_served"] = inp["recovery_risk_model"]
    _edit_payload(db, e, m)
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    _attack("Serve demoted model's value and re-sign", "key holder", "detected (replay: gate enforcement)",
            v["status"], v["status"] == "REPLAY_MISMATCH")
    assert v["status"] == "REPLAY_MISMATCH"


def test_A4_change_recommended_person_with_key(flow):
    db = flow["db"]
    e = next(x for x in flow["events"] if x.payload_json["data"]["recommended_employee_id"])
    snap = _snapshot(e)

    def m(p):
        d = p["data"]["receipt"]["decision"]
        others = [o["employee_id"] for o in p["data"]["receipt"]["outputs"] if o["employee_id"] != d["recommended_employee_id"]]
        d["recommended_employee_id"] = others[0]
    _edit_payload(db, e, m)
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    _attack("Change recommended person and re-sign", "key holder", "detected (replay: decision)",
            v["status"], v["status"] == "REPLAY_MISMATCH")
    assert v["status"] == "REPLAY_MISMATCH"


def test_A5_change_input_with_key(flow):
    db, e = flow["db"], flow["events"][0]
    snap = _snapshot(e)
    _edit_payload(db, e, lambda p: p["data"]["receipt"]["inputs"][0].update(skill=p["data"]["receipt"]["inputs"][0]["skill"] + 20))
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    _attack("Change a candidate input and re-sign", "key holder", "detected (replay: candidate)",
            v["status"], v["status"] == "REPLAY_MISMATCH")
    assert v["status"] == "REPLAY_MISMATCH"


def test_A6_change_rule_threshold_with_key(flow):
    db, e = flow["db"], flow["events"][0]
    snap = _snapshot(e)

    def m(p):
        r1 = next(r for r in p["data"]["receipt"]["policy"]["rules"] if r["code"] == "R1")
        r1["threshold"] = 95.0
    _edit_payload(db, e, m)
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    _attack("Change a rule threshold and re-sign", "key holder", "detected (replay: rules hash)",
            v["status"], v["status"] == "REPLAY_MISMATCH")
    assert v["status"] == "REPLAY_MISMATCH"


def test_A7_consistent_fabrication_with_key_is_not_detected(flow):
    """Documented limitation: a key holder who rewrites inputs, outputs and decision CONSISTENTLY
    (i.e. re-runs the scorer on fabricated inputs) produces a receipt that replays. Only an external
    anchor of the original hash would expose this."""
    db, e = flow["db"], flow["events"][0]
    snap = _snapshot(e)
    from app.ai import assignment_engine as ae, rules_engine

    def m(p):
        r = p["data"]["receipt"]
        r["inputs"][0]["skill"] = min(100, r["inputs"][0]["skill"] + 20)
        rules = rules_engine.rules_from_snapshot(r["policy"]["rules"])
        rescored = [ae.score_from_inputs(i, r["task"], rules, r["context"].get("emergency_mode", False)) for i in r["inputs"]]
        r["outputs"] = [dr._output_view(s) for s in rescored]
        ranked = ae.rank_candidates(rescored)
        top5 = (ranked["eligible"] + ranked["rule_excluded"])[:5]
        best = ranked["best"]
        final = best
        if best is not None:
            alt = ae.concurrent_absence_adjustment(top5, best, r["context"].get("prior_picks_of_top", 0))
            final = alt or best
        r["decision"].update(no_suitable_candidate=best is None,
                             recommended_employee_id=final["employee_id"] if final else None,
                             confidence=final["overall"] if final else None,
                             confidence_label=(rules_engine.confidence_label_with_rules(final["overall"], rules)
                                               if final else "Insufficient Confidence"),
                             top5_employee_ids=[c["employee_id"] for c in top5])
    _edit_payload(db, e, m)
    _resign_from(db, e)
    v = dr.verify_decision(db, e.event_id)
    _restore(db, e, *snap)
    _attack("Consistent fabrication of inputs+outputs, re-signed", "key holder",
            "NOT detected (limitation; needs external anchor)", v["status"], v["status"] != "VERIFIED")
    assert v["status"] == "VERIFIED"


def test_A8_edit_unhashed_columns(flow):
    db, e = flow["db"], flow["events"][0]
    old_actor, old_ts = e.actor_id, e.created_at
    e.actor_id = "manager"
    db.commit()
    st1 = audit_ledger.verify_event(db, e.event_id)["status"]
    e.actor_id = old_actor
    e.created_at = old_ts - timedelta(days=3)
    db.commit()
    st2 = audit_ledger.verify_event(db, e.event_id)["status"]
    e.created_at = old_ts
    db.commit()
    _attack("Edit actor column", "DB write access", "detected (columns)", st1, st1 == "COLUMNS_ALTERED")
    _attack("Back-date timestamp column", "DB write access", "detected (columns)", st2, st2 == "COLUMNS_ALTERED")
    assert st1 == st2 == "COLUMNS_ALTERED"
    assert audit_ledger.verify_event(db, e.event_id)["status"] == "VALID"


def test_A9_code_change_is_reported(flow):
    r = copy.deepcopy(flow["events"][0].payload_json["data"]["receipt"])
    r["code_fingerprint"]["combined"] = "0" * 64
    rep = dr.replay(r)
    _attack("Scoring code changed since decision", "developer", "reported (fingerprint)",
            f"{rep['status']}, fingerprint_matches={rep['code_fingerprint_matches']}",
            rep["code_fingerprint_matches"] is False)
    assert rep["code_fingerprint_matches"] is False


def test_zz_write_report(flow):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    gate = flow["events"][0].payload_json["data"]["receipt"]["gates"]["risk"]["record"]
    with open(REPORT, "w") as f:
        json.dump({"dataset": "D1 (synthetic seeded organisation)", "absent_employee_id": flow["absent"],
                   "decisions_recorded": len(flow["events"]), "decisions_verified": flow.get("verified"),
                   "replay_checks_run": flow.get("checks"),
                   "decisions_with_concurrent_absence_context": flow.get("adjusted"),
                   "risk_gate_status": gate["status"], "attacks": _ATTACKS}, f, indent=2, default=str)
    assert audit_ledger.verify_full_chain(flow["db"])["status"] == "VALID"
