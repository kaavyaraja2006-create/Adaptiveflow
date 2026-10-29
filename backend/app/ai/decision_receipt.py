"""
Replayable, verifiable decision receipts.

Every allocation recommendation produces a receipt that is stored inside the
signed, hash-chained ledger payload. The receipt contains everything needed
to RECOMPUTE the decision without the application database:

  * the task fields the score depends on,
  * every scored candidate's raw inputs (skill, history, load, the recovery
    risk that was served AND the baseline/model values the gate chose between),
  * the rule set in force (snapshot + hash),
  * the full baseline-gate records in force (+ hashes),
  * the scoring constants and a fingerprint of the scoring code,
  * the outputs (every candidate's score, breakdown and fired rules) and the
    final decision.

`replay(receipt)` re-runs the same pure functions the live system uses
(rules_engine.evaluate_with_rules, assignment_engine.score_from_inputs,
rank_candidates, concurrent_absence_adjustment) and checks that:

  1. the served recovery risk is the one the gate allowed (model if promoted,
     deterministic baseline otherwise)        -> a demoted model cannot have
                                                 influenced the decision;
  2. the stored rule snapshot and gate records match their hashes;
  3. every candidate's score, breakdown, exclusion and fired rules recompute;
  4. the ranking, viability cutoff and any concurrent-absence adjustment
     reproduce the recorded recommendation and confidence label.

`verify_decision(db, event_id)` combines this with the ledger's own hash,
signature and chain checks, so an auditor can confirm both that the record
is untampered AND that the decision follows from what it records.

Scope: replay proves internal consistency of the decision with its recorded
inputs under the recorded rules and gate. It does not prove the inputs were
correct when read from the database, and it is only exact while the scoring
code is unchanged (the receipt reports whether the code fingerprint matches).
"""
import hashlib
import json
import os
from typing import List, Optional

from . import assignment_engine as ae
from . import baseline_gate, rules_engine

RECEIPT_VERSION = "decision-receipt-v1"
_HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINT_FILES = ["assignment_engine.py", "rules_engine.py", "baseline_gate.py",
                     "decision_receipt.py", "workload_engine.py"]


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def code_fingerprint() -> dict:
    files = {}
    for name in FINGERPRINT_FILES:
        with open(os.path.join(_HERE, name), "rb") as f:
            files[name] = hashlib.sha256(f.read()).hexdigest()
    return {"files": files, "combined": sha256_json(files)}


def scoring_constants() -> dict:
    return {"score_version": ae.SCORE_VERSION, "positive_weights": ae.POSITIVE_WEIGHTS,
            "workload_lambda": ae.WORKLOAD_LAMBDA, "workload_load_weight": ae.WORKLOAD_LOAD_WEIGHT,
            "workload_risk_weight": ae.WORKLOAD_RISK_WEIGHT, "transfer_weight": ae.TRANSFER_WEIGHT,
            "reassignment_fairness_step": ae.REASSIGNMENT_FAIRNESS_STEP, "urgency_map": ae.URGENCY_MAP,
            "min_viable_score": ae.MIN_VIABLE_SCORE, "min_capacity": ae.MIN_CAPACITY}


def current_gate_records() -> dict:
    """Full gate records currently in force for each gated component (None if absent)."""
    from . import bigru, forecasting_engine
    out = {}
    for comp, status in (("risk", bigru.get_status()), ("duration", forecasting_engine.get_status())):
        trained = status.get("status") == "trained"
        rec = status.get("gate") if trained else None
        seg = status.get("segment_gate") if trained else None
        out[comp] = {"record": rec, "record_hash": sha256_json(rec) if rec else None,
                     "segment_record": seg, "segment_record_hash": sha256_json(seg) if seg else None,
                     "model_status": status.get("status")}
    return out


def _output_view(s: dict) -> dict:
    return {"employee_id": s["employee_id"], "overall": s["overall"], "excluded_by_rule": s["excluded_by_rule"],
            "exclusion_reason": s["exclusion_reason"], "applied_rule_ids": s["applied_rule_ids"],
            "requires_manager_review": s["requires_manager_review"], "score_breakdown": s["score_breakdown"],
            "transfer_cost": s["transfer_cost"]}


def build_receipt(db, *, task: dict, inputs: List[dict], scored: List[dict], result: dict,
                  opted_out_ids: List[int], context: dict) -> dict:
    snapshot = rules_engine.rules_snapshot(db)
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "task": task,
        "context": dict(context),
        "policy": {"rules": snapshot, "rules_hash": sha256_json(snapshot)},
        "scoring": scoring_constants(),
        "gates": current_gate_records(),
        "code_fingerprint": code_fingerprint(),
        "inputs": inputs,
        "outputs": [_output_view(s) for s in scored],
        "opted_out_employee_ids": list(opted_out_ids),
        "decision": {},
    }
    update_decision(receipt, result)
    return receipt


def update_decision(receipt: dict, result: dict) -> None:
    top5 = result.get("candidates") or []
    receipt["decision"] = {
        "no_suitable_candidate": bool(result["no_suitable_candidate"]),
        "recommended_employee_id": result.get("recommended_employee_id"),
        "confidence": result.get("confidence"),
        "confidence_label": result.get("confidence_label"),
        "top5_employee_ids": [c["employee_id"] for c in top5],
    }


def receipt_hash(receipt: dict) -> str:
    return sha256_json(receipt)


# ------------------------------------------------------------------ replay
def _check(checks: list, mismatches: list, name: str, ok: bool, detail=None):
    checks.append(name)
    if not ok:
        mismatches.append({"check": name, "detail": detail})


def replay(receipt: dict) -> dict:
    checks, mismatches = [], []
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        return {"status": "REPLAY_UNSUPPORTED", "reason": f"unknown receipt version {receipt.get('receipt_version')}"}

    # 2. integrity of embedded policy and gate records
    snap = receipt["policy"]["rules"]
    _check(checks, mismatches, "rules_hash", sha256_json(snap) == receipt["policy"]["rules_hash"])
    for comp, g in receipt["gates"].items():
        rec = g.get("record")
        _check(checks, mismatches, f"gate_hash:{comp}",
               (rec is None and g.get("record_hash") is None) or (rec is not None and sha256_json(rec) == g.get("record_hash")))
        seg = g.get("segment_record")
        _check(checks, mismatches, f"segment_gate_hash:{comp}",
               (seg is None and g.get("segment_record_hash") is None)
               or (seg is not None and sha256_json(seg) == g.get("segment_record_hash")))
        if seg is not None and rec is not None:
            # the segment record must agree with the global record it was built from
            g_status = rec.get("status")
            ok = seg.get("global_status") == g_status and all(
                r.get("status") == g_status for r in (seg.get("segments") or {}).values()
                if r.get("verdict_source") == "inherited_global")
            _check(checks, mismatches, f"segment_gate_consistent:{comp}", ok,
                   {"global_status": g_status, "segment_global_status": seg.get("global_status")})
    _check(checks, mismatches, "scoring_constants", receipt["scoring"] == scoring_constants(),
           "scoring constants differ from the running code")

    # 1. gate enforcement: the served recovery risk is the one the gate allowed
    risk_gate = (receipt["gates"].get("risk") or {}).get("record")
    risk_seg = (receipt["gates"].get("risk") or {}).get("segment_record")
    any_promoted = False
    for inp in receipt["inputs"]:
        model_v, base_v, served = inp["recovery_risk_model"], inp["recovery_risk_baseline"], inp["recovery_risk_served"]
        verdict = baseline_gate.verdict_for("risk", risk_gate, risk_seg, inp.get("recovery_risk_today"))
        promoted = verdict["status"] == baseline_gate.PROMOTED
        any_promoted = any_promoted or (promoted and model_v is not None)
        if model_v is None:
            ok = base_v is None  # no model ran; served value is the deterministic score itself
        elif promoted:
            ok = served == model_v
        else:
            ok = served == base_v
        _check(checks, mismatches, f"gate_enforced:{inp['employee_id']}", ok,
               {"served": served, "model": model_v, "baseline": base_v, "verdict": verdict})
        if model_v is not None and inp.get("recovery_risk_gate_status") is not None:
            _check(checks, mismatches, f"gate_status_consistent:{inp['employee_id']}",
                   inp["recovery_risk_gate_status"] == verdict["status"],
                   {"recorded": inp["recovery_risk_gate_status"], "recomputed": verdict})

    # 3. per-candidate recomputation
    rules = rules_engine.rules_from_snapshot(snap)
    task, ctx = receipt["task"], receipt["context"]
    rescored = [ae.score_from_inputs(inp, task, rules, ctx.get("emergency_mode", False))
                for inp in receipt["inputs"]]
    recorded = {o["employee_id"]: o for o in receipt["outputs"]}
    _check(checks, mismatches, "candidate_set",
           [o["employee_id"] for o in receipt["outputs"]] == [i["employee_id"] for i in receipt["inputs"]])
    for s in rescored:
        o = recorded.get(s["employee_id"])
        view = _output_view(s)
        _check(checks, mismatches, f"candidate:{s['employee_id']}", o == view,
               None if o == view else {"recorded": o, "recomputed": view})

    # 4. ranking, viability, adjustment, label
    ranked = ae.rank_candidates(rescored)
    top5 = (ranked["eligible"] + ranked["rule_excluded"])[:5]
    best = ranked["best"]
    final = best
    if best is not None:
        alt = ae.concurrent_absence_adjustment(top5, best, ctx.get("prior_picks_of_top", 0))
        if alt is not None:
            final = alt
    dec = receipt["decision"]
    expect = {
        "no_suitable_candidate": best is None,
        "recommended_employee_id": final["employee_id"] if final else None,
        "confidence": final["overall"] if final else None,
        "confidence_label": (rules_engine.confidence_label_with_rules(final["overall"], rules)
                             if final else "Insufficient Confidence"),
        "top5_employee_ids": [c["employee_id"] for c in top5],
    }
    for k, v in expect.items():
        _check(checks, mismatches, f"decision:{k}", dec.get(k) == v, {"recorded": dec.get(k), "recomputed": v})

    fp_now = code_fingerprint()["combined"]
    return {"status": "REPLAY_MATCH" if not mismatches else "REPLAY_MISMATCH",
            "checks_run": len(checks), "mismatches": mismatches,
            "code_fingerprint_matches": fp_now == receipt["code_fingerprint"]["combined"],
            "risk_model_used_for_any_candidate": any_promoted}


def verify_decision(db, event_id: str) -> dict:
    """Ledger integrity (hash, signature, chain, envelope/column consistency) + receipt replay."""
    from .. import models
    from . import audit_ledger
    ledger = audit_ledger.verify_event(db, event_id)
    event = db.query(models.AuditEvent).filter(models.AuditEvent.event_id == event_id).first()
    receipt = ((event.payload_json or {}).get("data") or {}).get("receipt") if event else None
    if ledger.get("status") != "VALID":
        return {"status": ledger.get("status"), "ledger": ledger, "replay": None}
    if receipt is None:
        return {"status": "NO_RECEIPT", "ledger": ledger, "replay": None,
                "reason": "This event carries no decision receipt (e.g. written by an older version)."}
    rep = replay(receipt)
    status = "VERIFIED" if rep["status"] == "REPLAY_MATCH" else rep["status"]
    return {"status": status, "ledger": ledger, "replay": rep, "receipt_hash": receipt_hash(receipt)}
