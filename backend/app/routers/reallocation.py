"""
Phase 1-2 endpoints: human-aware allocation analysis, workload/recovery-risk
profiles, and the manager-configurable symbolic rule set. Additive to the
existing /api/absence/* flow (which still works unchanged) — these give
direct, spec-shaped access to the same underlying engine.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..ai import assignment_engine, workload_engine, rules_engine, data_quality

router = APIRouter(prefix="/api", tags=["allocation"])


@router.post("/ai/reallocation/analyze", response_model=schemas.ReassignmentResult)
def analyze_reallocation(payload: schemas.ReallocationAnalyzeRequest, db: Session = Depends(get_db)):
    issue = db.query(models.Issue).get(payload.task_id)
    if not issue:
        raise HTTPException(404, "Task not found")
    return assignment_engine.recommend_for_issue(db, issue, emergency_mode=payload.emergency_mode)


@router.get("/ai/reallocation/{task_id}/candidates", response_model=schemas.ReassignmentResult)
def reallocation_candidates(task_id: int, db: Session = Depends(get_db)):
    issue = db.query(models.Issue).get(task_id)
    if not issue:
        raise HTTPException(404, "Task not found")
    return assignment_engine.recommend_for_issue(db, issue)


@router.get("/employees/{employee_id}/workload-profile", response_model=schemas.WorkloadProfileOut)
def employee_workload_profile(employee_id: int, db: Session = Depends(get_db)):
    profile = workload_engine.workload_profile_by_id(db, employee_id)
    if not profile:
        raise HTTPException(404, "Employee not found")
    return profile.__dict__


@router.get("/employees/{employee_id}/recovery-risk")
def employee_recovery_risk(employee_id: int, db: Session = Depends(get_db)):
    profile = workload_engine.workload_profile_by_id(db, employee_id)
    if not profile:
        raise HTTPException(404, "Employee not found")
    return {
        "employee_id": profile.employee_id,
        "name": profile.name,
        "opted_out": profile.opted_out,
        "recovery_risk": profile.recovery_risk,
        "risk_band": profile.risk_band,
        "model_status": profile.model_status,
        "model_version": profile.model_version,
        "stale_data_flags": profile.stale_data_flags,
        "note": "Sustainable Workload / Recovery Risk — a work-pattern signal only, never a medical or "
                "emotional assessment. Computed from sprint load, task-switching, and after-hours activity.",
    }


@router.get("/allocation-rules", response_model=list[schemas.AllocationRuleOut])
def list_allocation_rules(db: Session = Depends(get_db)):
    rules_engine.ensure_default_rules(db)
    return db.query(models.AllocationRule).order_by(models.AllocationRule.code).all()


@router.patch("/allocation-rules/{code}", response_model=schemas.AllocationRuleOut)
def update_allocation_rule(code: str, payload: schemas.AllocationRuleUpdate, db: Session = Depends(get_db)):
    rule = db.query(models.AllocationRule).filter(models.AllocationRule.code == code).first()
    if not rule:
        raise HTTPException(404, "Rule not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(rule, k, v)
    db.commit()
    db.refresh(rule)
    return rule


@router.get("/data-quality/scan")
def data_quality_scan(db: Session = Depends(get_db)):
    return data_quality.run_project_data_quality_scan(db)


@router.get("/ai/reallocation/{task_id}/simulate-candidates")
def simulate_reallocation_candidates(task_id: int, db: Session = Depends(get_db)):
    """Connects the digital twin to the allocation engine: re-simulates the
    sprint with each top candidate actually assigned, rather than only
    reporting their allocation score."""
    from ..ai import digital_twin
    issue = db.query(models.Issue).get(task_id)
    if not issue:
        raise HTTPException(404, "Task not found")
    from ..ai import audit_ledger, decision_receipt
    rec = assignment_engine.recommend_for_issue(db, issue)
    candidates = [c for c in rec["candidates"] if not c.get("excluded_by_rule")]
    sims = digital_twin.simulate_reassignment_candidates(db, issue, candidates)
    # bind the simulated consequences to the recommendation they were computed for
    rec_ev = audit_ledger.latest_event(db, task_id, ["REALLOCATION_RECOMMENDED"])
    ev = audit_ledger.record_event(
        db, event_type="CANDIDATE_SIMULATION_COMPLETED",
        payload={"task_id": task_id, "candidates": sims,
                 "receipt_hash_of_this_ranking": decision_receipt.receipt_hash(rec["receipt"]),
                 "recommendation_event_id": rec_ev.event_id if rec_ev else None,
                 "recommendation_event_hash": rec_ev.event_hash if rec_ev else None},
        actor_id="system-ai", project_id=issue.project_id, task_id=task_id, model_version="digital-twin-v1.0",
    )
    return {"task_id": task_id, "task_title": issue.title, "candidates": sims, "simulation_event_id": ev.event_id}


@router.get("/ai/models/status")
def sequence_model_status():
    from ..ai import bigru
    return bigru.get_status()


@router.post("/ai/models/train")
def train_sequence_model(db: Session = Depends(get_db)):
    """Trains the BiGRU workload/recovery-risk model on real seeded activity
    sequences. Honest by construction: if there isn't enough sequence data
    yet, this returns status="insufficient_data" and trains nothing rather
    than faking a result — see ai/bigru.py."""
    from ..ai import bigru
    sequences = workload_engine.build_sequences_for_training(db)
    return bigru.train(sequences)
