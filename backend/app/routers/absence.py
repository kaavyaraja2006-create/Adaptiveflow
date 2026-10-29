from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from .. import models, schemas
from ..database import get_db
from ..ai import assignment_engine, audit_ledger, decision_receipt

router = APIRouter(prefix="/api/absence", tags=["absence"])


@router.post("/mark", response_model=list[schemas.ReassignmentResult])
def mark_absent(payload: schemas.AbsenceRequest, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(payload.employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")

    emp.status = "absent"
    db.add(models.Absence(employee_id=emp.id, status="absent"))
    db.commit()

    results = assignment_engine.handle_absence(db, emp.id)
    record_recommendations(db, results, emp.id, trigger="employee_absence")
    return results


def record_recommendations(db: Session, results: list, absent_employee_id: int, trigger: str) -> None:
    """Store each recommendation as PENDING (a manager must approve it via /reassign or reject it via
    /reject) and write a signed ledger event carrying its replayable decision receipt.
    Nothing is reassigned here: no code path in AdaptiveFlow reassigns a task without a manager decision."""
    for r in results:
        rec = models.Recommendation(
            kind="reassignment",
            title=(f"Reassign \u201c{r['task_title']}\u201d to {r['recommended_employee_name']}"
                   if not r["no_suitable_candidate"] else f"No suitable reassignment for \u201c{r['task_title']}\u201d"),
            reason=r["reason"] or "",
            evidence=f"employee_id={r.get('recommended_employee_id')} scored against {len(r['candidates'])} candidates",
            confidence=r["confidence"] or 0,
            related_issue_id=r["task_id"],
        )
        db.add(rec)

        issue = db.query(models.Issue).get(r["task_id"])
        audit_ledger.record_event(
            db, event_type="REALLOCATION_RECOMMENDED",
            payload={
                "task_id": r["task_id"], "task_title": r["task_title"],
                "recommended_employee_id": r.get("recommended_employee_id"),
                "confidence": r.get("confidence"), "confidence_label": r.get("confidence_label"),
                "no_suitable_candidate": r["no_suitable_candidate"],
                "candidate_count": len(r["candidates"]),
                "trigger": trigger, "absent_employee_id": absent_employee_id,
                # everything needed to recompute this decision offline (see ai/decision_receipt.py)
                "receipt": r["receipt"],
                "receipt_hash": decision_receipt.receipt_hash(r["receipt"]),
            },
            actor_id="system-ai", project_id=issue.project_id if issue else None, task_id=r["task_id"],
            model_version=assignment_engine.SCORE_VERSION,
            rule_ids=(r.get("candidates")[0]["applied_rule_ids"] if r.get("candidates") else []),
        )
        r["status"] = "pending_manager_approval"
    db.commit()


@router.post("/return/{employee_id}")
def return_from_absence(employee_id: int, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    emp.status = "available"
    open_absence = (
        db.query(models.Absence)
        .filter(models.Absence.employee_id == employee_id, models.Absence.ended_at.is_(None))
        .order_by(models.Absence.id.desc())
        .first()
    )
    if open_absence:
        open_absence.ended_at = datetime.utcnow()
    db.commit()
    return {"employee_id": employee_id, "status": "available"}


@router.post("/reassign/{issue_id}/{employee_id}")
def approve_reassignment(issue_id: int, employee_id: int, decision: str = "approved",
                          db: Session = Depends(get_db)):
    """`decision`: "approved" (manager accepted the top recommendation) or
    "modified" (manager picked a different candidate than the one
    recommended) — both are legitimate human-in-the-loop outcomes and are
    logged distinctly in the audit ledger, but both result in the same real
    action: the task IS reassigned, only after this explicit call."""
    issue = db.query(models.Issue).get(issue_id)
    employee = db.query(models.Employee).get(employee_id)
    if not issue or not employee:
        raise HTTPException(404, "Issue or employee not found")

    old_assignee_id = issue.assignee_id
    link = _decision_link(db, issue_id)
    recommended = link.get("recommended_employee_id")
    followed = (recommended == employee_id) if link.get("recommendation_event_id") else None
    # the event type follows what the manager actually did, not only what the client declared
    if followed is not None:
        event_type = "RECOMMENDATION_APPROVED" if followed else "RECOMMENDATION_MODIFIED"
    else:
        event_type = "RECOMMENDATION_APPROVED" if decision == "approved" else "RECOMMENDATION_MODIFIED"
    issue.assignee_id = employee_id
    db.add(models.DecisionHistoryEntry(
        recommendation=f"Reassign issue #{issue_id} ('{issue.title}') to employee_id={employee_id}",
        model="AbsenceRecoveryEngine v1",
        confidence=80,
        evidence=f"Previous assignee_id={old_assignee_id}",
        human_decision="Approved" if decision == "approved" else "Modified",
        outcome="Reassignment executed",
    ))

    audit_ledger.record_event(
        db, event_type=event_type,
        payload={"task_id": issue_id, "task_title": issue.title, "previous_assignee_id": old_assignee_id,
                 "manager_action": decision, "chosen_employee_id": employee_id,
                 "followed_recommendation": followed,
                 "declared_action_consistent": (followed is None) or (followed == (decision == "approved")),
                 **link},
        actor_id="manager", project_id=issue.project_id, task_id=issue_id,
    )
    audit_ledger.record_event(
        db, event_type="TASK_REASSIGNED",
        payload={"task_id": issue_id, "task_title": issue.title,
                 "previous_assignee_id": old_assignee_id, "new_assignee_id": employee_id},
        actor_id="manager", project_id=issue.project_id, task_id=issue_id,
    )
    db.commit()
    return {"issue_id": issue_id, "new_assignee_id": employee_id}


@router.post("/reject/{issue_id}")
def reject_reassignment(issue_id: int, reason: str = "", db: Session = Depends(get_db)):
    """Manager rejects the AI's recommendation outright — no reassignment
    happens, but the decision (and why) is still recorded, same as an
    approval would be. This is what makes the ledger a genuine decision
    record rather than only a log of actions taken."""
    issue = db.query(models.Issue).get(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")

    db.add(models.DecisionHistoryEntry(
        recommendation=f"Reassignment recommendation for issue #{issue_id} ('{issue.title}')",
        model="AbsenceRecoveryEngine v1", confidence=0,
        evidence=reason or "Manager rejected without a stated reason",
        human_decision="Rejected", outcome="No action taken",
    ))
    audit_ledger.record_event(
        db, event_type="RECOMMENDATION_REJECTED",
        payload={"task_id": issue_id, "task_title": issue.title, "reason": reason, **_decision_link(db, issue_id)},
        actor_id="manager", project_id=issue.project_id, task_id=issue_id,
    )
    db.commit()
    return {"issue_id": issue_id, "status": "rejected"}


def _decision_link(db: Session, issue_id: int) -> dict:
    """Bind a manager decision to the exact recommendation (and candidate simulation, if one was run)
    it responds to, by event hash and receipt hash."""
    rec_ev = audit_ledger.latest_event(db, issue_id, ["REALLOCATION_RECOMMENDED"])
    if rec_ev is None:
        return {"recommendation_event_id": None}
    data = (rec_ev.payload_json or {}).get("data") or {}
    sim_ev = audit_ledger.latest_event(db, issue_id, ["CANDIDATE_SIMULATION_COMPLETED"])
    sim_ok = sim_ev is not None and sim_ev.id > rec_ev.id
    return {
        "recommendation_event_id": rec_ev.event_id,
        "recommendation_event_hash": rec_ev.event_hash,
        "receipt_hash": data.get("receipt_hash"),
        "recommended_employee_id": data.get("recommended_employee_id"),
        "simulation_event_id": sim_ev.event_id if sim_ok else None,
        "simulation_event_hash": sim_ev.event_hash if sim_ok else None,
    }
