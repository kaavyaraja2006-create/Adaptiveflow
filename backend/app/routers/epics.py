from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/epics", tags=["epics"])


@router.get("", response_model=list[schemas.EpicOut])
def list_epics(project_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Epic)
    if project_id:
        q = q.filter(models.Epic.project_id == project_id)
    return q.all()


@router.post("", response_model=schemas.EpicOut)
def create_epic(payload: schemas.EpicIn, db: Session = Depends(get_db)):
    epic = models.Epic(title=payload.title, description=payload.description, project_id=payload.project_id, color=payload.color)
    db.add(epic)
    db.commit()
    db.refresh(epic)
    project = db.query(models.Project).get(payload.project_id)
    epic.key = f"{project.key if project and project.key else 'EPIC'}-E{epic.id}"
    db.commit()
    db.refresh(epic)
    return epic


@router.get("/{epic_id}/progress", response_model=schemas.EpicProgress)
def epic_progress(epic_id: int, db: Session = Depends(get_db)):
    epic = db.query(models.Epic).get(epic_id)
    if not epic:
        raise HTTPException(404, "Epic not found")
    issues = db.query(models.Issue).filter(models.Issue.epic_id == epic_id).all()
    total = sum(i.story_points for i in issues)
    completed = sum(i.story_points for i in issues if i.status == "Done")
    pct = round(completed / total * 100) if total else 0
    return schemas.EpicProgress(
        epic=epic, total_points=total, completed_points=completed,
        remaining_points=total - completed, issue_count=len(issues), progress_pct=pct,
    )


@router.delete("/{epic_id}")
def delete_epic(epic_id: int, db: Session = Depends(get_db)):
    epic = db.query(models.Epic).get(epic_id)
    if not epic:
        raise HTTPException(404, "Epic not found")
    db.query(models.Issue).filter(models.Issue.epic_id == epic_id).update({"epic_id": None})
    db.delete(epic)
    db.commit()
    return {"deleted": True}
