from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..ai import biometric_engine, assignment_engine
from . import absence as absence_router

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


def _mark_present(db: Session, employee: models.Employee, method: str, match_distance: float | None):
    was_absent = employee.status == "absent"
    employee.status = "available"
    if was_absent:
        open_absence = (
            db.query(models.Absence)
            .filter(models.Absence.employee_id == employee.id, models.Absence.ended_at.is_(None))
            .order_by(models.Absence.id.desc())
            .first()
        )
        if open_absence:
            open_absence.ended_at = datetime.utcnow()
    db.add(models.Attendance(employee_id=employee.id, method=method, match_distance=match_distance))
    db.commit()


@router.post("/enroll/{employee_id}")
def enroll_face(employee_id: int, payload: schemas.FaceEnrollIn, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    if len(payload.descriptor) < 32:
        raise HTTPException(400, "Descriptor looks too short to be a real face embedding")
    emp.face_descriptor = payload.descriptor
    db.commit()
    return {"employee_id": employee_id, "enrolled": True}


@router.delete("/enroll/{employee_id}")
def remove_enrollment(employee_id: int, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    emp.face_descriptor = None
    db.commit()
    return {"employee_id": employee_id, "enrolled": False}


@router.post("/check-in/biometric", response_model=schemas.CheckInResult)
def check_in_biometric(payload: schemas.BiometricCheckInIn, db: Session = Depends(get_db)):
    emp, dist = biometric_engine.find_best_match(db, payload.descriptor)
    if not emp:
        return schemas.CheckInResult(
            matched=False, match_distance=dist,
            message="No confident match found. Try again or use manual check-in.",
        )
    _mark_present(db, emp, "biometric", dist)
    return schemas.CheckInResult(
        matched=True, employee_id=emp.id, employee_name=emp.name, match_distance=dist,
        message=f"Checked in {emp.name} via biometric match (distance {dist:.3f}).",
    )


@router.post("/check-in/manual", response_model=schemas.CheckInResult)
def check_in_manual(payload: schemas.ManualCheckInIn, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(payload.employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    _mark_present(db, emp, "manual", None)
    return schemas.CheckInResult(matched=True, employee_id=emp.id, employee_name=emp.name, message=f"Checked in {emp.name} manually.")


@router.post("/check-in/manual-bulk", response_model=list[schemas.CheckInResult])
def check_in_manual_bulk(payload: schemas.ManualBulkCheckInIn, db: Session = Depends(get_db)):
    results = []
    for employee_id in payload.employee_ids:
        emp = db.query(models.Employee).get(employee_id)
        if not emp:
            results.append(schemas.CheckInResult(matched=False, employee_id=employee_id, message="Employee not found"))
            continue
        _mark_present(db, emp, "manual", None)
        results.append(schemas.CheckInResult(matched=True, employee_id=emp.id, employee_name=emp.name, message=f"Checked in {emp.name}."))
    return results


@router.get("/today")
def attendance_today(db: Session = Depends(get_db)):
    today = date.today()
    records = (
        db.query(models.Attendance)
        .filter(models.Attendance.checked_in_at >= datetime(today.year, today.month, today.day))
        .all()
    )
    checked_in_ids = {r.employee_id for r in records}
    employees = db.query(models.Employee).all()
    out = []
    for e in employees:
        record = next((r for r in records if r.employee_id == e.id), None)
        out.append({
            "employee_id": e.id,
            "name": e.name,
            "checked_in": e.id in checked_in_ids,
            "method": record.method if record else None,
            "status": e.status,
            "face_enrolled": e.face_enrolled,
        })
    return out


@router.post("/submit-checkin")
def submit_checkin(payload: schemas.SubmitCheckInIn, db: Session = Depends(get_db)):
    """
    One-shot end-of-window attendance submission: whoever is passed as
    present gets checked in; everyone else is marked absent and their open
    tasks are run through the allocation pipeline. Each result is stored as
    a PENDING recommendation with a signed, replayable ledger receipt. No task
    is reassigned here: a manager must approve (/api/absence/reassign) or
    reject (/api/absence/reject) every recommendation. (Earlier versions
    auto-approved the top candidate; that path has been removed.)
    """
    today = date.today()
    today_start = datetime(today.year, today.month, today.day)
    already_checked_ids = {
        r.employee_id for r in
        db.query(models.Attendance).filter(models.Attendance.checked_in_at >= today_start).all()
    }
    present_ids = set(payload.present_employee_ids) | already_checked_ids

    checked_in_now = []
    newly_absent = []
    reassignments = []
    batch_tentative_load: dict = {}  # shared across every absent person processed in this same submission

    all_employees = db.query(models.Employee).all()

    # Pass 1: check in the present, and mark everyone else absent FIRST (before
    # any reassignment runs) so that someone who is themselves newly absent
    # can never be recommended as a candidate for someone else's task just
    # because their own absence hadn't been committed yet.
    for emp in all_employees:
        if emp.id in present_ids:
            if emp.id not in already_checked_ids:
                _mark_present(db, emp, "manual", None)
                checked_in_now.append({"employee_id": emp.id, "name": emp.name})
            continue
        if emp.status != "absent":
            emp.status = "absent"
            db.add(models.Absence(employee_id=emp.id, status="absent"))
            newly_absent.append({"employee_id": emp.id, "name": emp.name})
    db.commit()

    # Pass 2: now that every absence in this batch is committed, run
    # reassignment for each newly-absent person against the correct pool.
    for entry in newly_absent:
        emp = db.query(models.Employee).get(entry["employee_id"])
        recs = assignment_engine.handle_absence(db, emp.id, tentative_load=batch_tentative_load)
        absence_router.record_recommendations(db, recs, emp.id, trigger="checkin_absence")
        for r in recs:
            r.pop("receipt", None)   # stored in the ledger; not needed in the response
            reassignments.append({**r, "employee_absent": emp.name})

    return {
        "checked_in": checked_in_now,
        "auto_marked_absent": newly_absent,
        "reassignments": reassignments,
    }
