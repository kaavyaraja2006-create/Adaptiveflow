"""Replay stress test through the LIVE code path (database -> recommendation -> signed ledger -> verifier).

Each run builds a fresh synthetic organization and randomizes it: employee success records, utilization, opt-outs,
issue story points and priorities, the rule thresholds and enabled flags, and the BiGRU gate verdicts (global and
per segment, promoted or demoted). It then marks random employees absent, verifies every recorded decision, and
applies one random key-holder manipulation to each decision (re-signed) to measure detection.

    python analysis/replay_stress.py --runs 30
"""
import argparse
import copy
import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tests")
import conftest  # noqa: E402,F401  (temporary model store and signing key)
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from app.database import Base  # noqa: E402
from app import models, seed as seed_module, schemas  # noqa: E402
from app.ai import rules_engine, bigru, workload_engine, audit_ledger, decision_receipt as dr, baseline_gate as bg  # noqa: E402
from app.routers import absence  # noqa: E402

MUTATIONS = ["input_skill", "input_load", "served_risk", "rule_threshold", "gate_global", "gate_segment",
             "recommended_person", "confidence"]


def resign_from(db, ev):
    later = db.query(models.AuditEvent).filter(models.AuditEvent.id >= ev.id).order_by(models.AuditEvent.id).all()
    prev = ev.previous_hash
    for e in later:
        e.previous_hash = prev
        e.event_hash = audit_ledger.compute_event_hash(e.payload_json, prev)
        e.signature = audit_ledger.sign_hash(e.event_hash)
        prev = e.event_hash
    db.commit()


def mutate(p, kind, rnd):
    r = p["data"]["receipt"]
    if kind == "input_skill":
        i = rnd.choice(r["inputs"]); i["skill"] = i["skill"] + rnd.choice([-15, -5, 5, 15])
    elif kind == "input_load":
        i = rnd.choice(r["inputs"]); i["load_ratio"] = round(i["load_ratio"] + rnd.choice([-0.3, 0.3]), 3)
    elif kind == "served_risk":
        cands = [i for i in r["inputs"] if i["recovery_risk_model"] is not None
                 and i["recovery_risk_model"] != i["recovery_risk_baseline"]]
        if not cands:
            return False
        i = rnd.choice(cands)
        i["recovery_risk_served"] = (i["recovery_risk_baseline"] if i["recovery_risk_served"] == i["recovery_risk_model"]
                                     else i["recovery_risk_model"])
    elif kind == "rule_threshold":
        rule = rnd.choice([x for x in r["policy"]["rules"] if x["threshold"] is not None])
        rule["threshold"] = rule["threshold"] * rnd.choice([0.8, 1.2])
        r["policy"]["rules_hash"] = dr.sha256_json(r["policy"]["rules"])        # attacker fixes the hash
    elif kind == "gate_global":
        g = r["gates"]["risk"]
        if not g.get("record"):
            return False
        g["record"]["status"] = bg.DEMOTED if g["record"]["status"] == bg.PROMOTED else bg.PROMOTED
        g["record_hash"] = dr.sha256_json(g["record"])
    elif kind == "gate_segment":
        g = r["gates"]["risk"]
        if not g.get("segment_record"):
            return False
        seg = rnd.choice(list(g["segment_record"]["segments"].values()))
        seg["status"] = bg.DEMOTED if seg["status"] == bg.PROMOTED else bg.PROMOTED
        g["segment_record_hash"] = dr.sha256_json(g["segment_record"])
    elif kind == "recommended_person":
        d = r["decision"]
        others = [o["employee_id"] for o in r["outputs"] if o["employee_id"] != d["recommended_employee_id"]]
        if not others:
            return False
        d["recommended_employee_id"] = rnd.choice(others)
    elif kind == "confidence":
        d = r["decision"]
        if d["confidence"] is None:
            return False
        d["confidence"] = d["confidence"] + rnd.choice([-7, 7])
    return True


def one_run(seed, n_absent):
    rnd = random.Random(seed)
    eng = create_engine(f"sqlite:///{tempfile.mkdtemp()}/s.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    seed_module.seed(db)
    rules_engine.ensure_default_rules(db)
    for h in db.query(models.WorkHistoryItem).all():
        h.success = rnd.random() < rnd.uniform(0.4, 0.95)
    for e in db.query(models.Employee).all():
        e.workload_opt_out = rnd.random() < 0.1
    for i in db.query(models.Issue).filter(models.Issue.status != "Done").all():
        i.story_points = rnd.choice([1, 2, 3, 5, 8, 13])
        i.priority = rnd.choice(["Critical", "High", "Medium", "Low"])
    for r in db.query(models.AllocationRule).all():
        if r.threshold is not None:
            r.threshold = r.threshold * rnd.uniform(0.85, 1.15)
        r.enabled = rnd.random() > 0.15
    db.commit()
    meta = bigru.train(workload_engine.build_sequences_for_training(db))
    if meta.get("status") == "trained":                          # randomize the gate verdicts the system obeys
        g_status = rnd.choice([bg.PROMOTED, bg.DEMOTED])
        meta["gate"]["status"] = g_status
        meta["segment_gate"]["global_status"] = g_status
        for s in meta["segment_gate"]["segments"].values():
            s["status"], s["verdict_source"] = rnd.choice([bg.PROMOTED, bg.DEMOTED]), "segment_test_holm"
        json.dump(meta, open(bigru.META_PATH, "w"))
    emps = [e.id for e in db.query(models.Employee).all()]
    for eid in rnd.sample(emps, n_absent):
        if db.query(models.Employee).get(eid).status != "absent":
            absence.mark_absent(schemas.AbsenceRequest(employee_id=eid), db)
    db.expire_all()
    evs = db.query(models.AuditEvent).filter(models.AuditEvent.event_type == "REALLOCATION_RECOMMENDED").all()
    verified, promoted_used, detect = 0, 0, {k: [0, 0] for k in MUTATIONS}
    for e in evs:
        v = dr.verify_decision(db, e.event_id)
        verified += v["status"] == "VERIFIED"
        promoted_used += bool(v.get("replay") and v["replay"].get("risk_model_used_for_any_candidate"))
    for e in evs:
        kind = rnd.choice(MUTATIONS)
        snap = (copy.deepcopy(e.payload_json), e.previous_hash, e.event_hash, e.signature)
        p = copy.deepcopy(e.payload_json)
        if not mutate(p, kind, rnd):
            continue
        e.payload_json = p; db.commit(); resign_from(db, e)
        detect[kind][1] += 1
        detect[kind][0] += dr.verify_decision(db, e.event_id)["status"] != "VERIFIED"
        e.payload_json, e.previous_hash, e.event_hash, e.signature = snap; db.commit(); resign_from(db, e)
    chain = audit_ledger.verify_full_chain(db)["status"]
    db.close()
    return len(evs), verified, promoted_used, detect, chain


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=30); ap.add_argument("--absent", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    tot = {"decisions": 0, "verified": 0, "with_promoted_model": 0, "chains_valid": 0,
           "detection": {k: [0, 0] for k in MUTATIONS}}
    for s in range(a.runs):
        n, ver, prom, det, chain = one_run(1000 + s, a.absent)
        tot["decisions"] += n; tot["verified"] += ver; tot["with_promoted_model"] += prom
        tot["chains_valid"] += chain == "VALID"
        for k, (x, y) in det.items():
            tot["detection"][k][0] += x; tot["detection"][k][1] += y
        print(f"run {s + 1}/{a.runs}: {n} decisions, {ver} verified, {prom} used a promoted model", flush=True)
    tot["runs"] = a.runs
    tot["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    json.dump(tot, open("results/replay_stress.json", "w"), indent=1)
    print(json.dumps(tot, indent=1))
