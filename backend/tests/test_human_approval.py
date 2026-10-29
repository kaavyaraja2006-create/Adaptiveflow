"""No code path reassigns a task without a manager decision (paper design principle 'human in the loop')."""
import os
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models, schemas
from app.ai import audit_ledger, decision_receipt as dr
from app.routers import absence, attendance


def _fresh_db():
    from app.database import Base
    from app import seed as seed_module
    from app.ai import rules_engine
    eng = create_engine(f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'h.db')}",
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    seed_module.seed(db)
    rules_engine.ensure_default_rules(db)
    return db


def test_checkin_creates_pending_recommendations_only():
    db = _fresh_db()
    emps = db.query(models.Employee).all()
    open_count = {e.id: db.query(models.Issue).filter(models.Issue.assignee_id == e.id,
                                                      models.Issue.status != "Done").count() for e in emps}
    absent = max(open_count, key=open_count.get)
    before = {i.id: i.assignee_id for i in db.query(models.Issue).all()}

    present = [e.id for e in emps if e.id != absent]
    res = attendance.submit_checkin(schemas.SubmitCheckInIn(present_employee_ids=present), db)
    db.expire_all()

    after = {i.id: i.assignee_id for i in db.query(models.Issue).all()}
    assert after == before, "check-in must not reassign anything by itself"
    assert res["reassignments"] and all(r["status"] == "pending_manager_approval" for r in res["reassignments"])
    assert not db.query(models.DecisionHistoryEntry).filter(
        models.DecisionHistoryEntry.human_decision.like("Auto%")).count()

    events = db.query(models.AuditEvent).filter(models.AuditEvent.event_type == "REALLOCATION_RECOMMENDED").all()
    assert len(events) == len(res["reassignments"])
    assert all(e.payload_json["data"]["trigger"] == "checkin_absence" for e in events)
    assert all(dr.verify_decision(db, e.event_id)["status"] == "VERIFIED" for e in events)

    # only the manager's explicit approval changes the assignee
    r = next(x for x in res["reassignments"] if not x["no_suitable_candidate"])
    absence.approve_reassignment(r["task_id"], r["recommended_employee_id"], db=db)
    db.expire_all()
    assert db.query(models.Issue).get(r["task_id"]).assignee_id == r["recommended_employee_id"]
    ev = audit_ledger.latest_event(db, r["task_id"], ["RECOMMENDATION_APPROVED", "RECOMMENDATION_MODIFIED"])
    assert ev.event_type == "RECOMMENDATION_APPROVED" and ev.payload_json["data"]["followed_recommendation"] is True
    assert audit_ledger.verify_full_chain(db)["status"] == "VALID"
    db.close()


def test_no_unapproved_assignment_writes_in_routers():
    """Static guard: assignee changes appear only in the manager-approval route."""
    import pathlib
    import re
    here = pathlib.Path(__file__).resolve().parents[1] / "app" / "routers"
    hits = [(p.name, i) for p in here.glob("*.py") for i, line in enumerate(p.read_text().splitlines(), 1)
            if re.search(r"\.assignee_id\s*=(?!=)", line)]
    assert [h[0] for h in hits] == ["absence.py"], hits
