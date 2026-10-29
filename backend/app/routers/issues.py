from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/issues", tags=["issues"])

TRACKED_FIELDS = ["status", "priority", "assignee_id", "sprint_id", "epic_id", "blocked"]


def _log_activity(db: Session, issue_id: int, field: str, old_value, new_value, actor: str = "System"):
    if old_value == new_value:
        return
    db.add(models.ActivityLogEntry(
        issue_id=issue_id, actor=actor, field=field,
        old_value=str(old_value) if old_value is not None else None,
        new_value=str(new_value) if new_value is not None else None,
    ))


def _generate_key(db: Session, project_id: int, issue_id: int) -> str:
    project = db.query(models.Project).get(project_id) if project_id else None
    prefix = project.key if project and project.key else "ISS"
    return f"{prefix}-{issue_id}"


@router.get("", response_model=list[schemas.IssueOut])
def list_issues(
    project_id: int | None = None,
    sprint_id: int | None = None,
    status: str | None = None,
    assignee_id: int | None = None,
    epic_id: int | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(models.Issue)
    if project_id:
        q = q.filter(models.Issue.project_id == project_id)
    if sprint_id:
        q = q.filter(models.Issue.sprint_id == sprint_id)
    if status:
        q = q.filter(models.Issue.status == status)
    if assignee_id:
        q = q.filter(models.Issue.assignee_id == assignee_id)
    if epic_id:
        q = q.filter(models.Issue.epic_id == epic_id)
    return q.order_by(models.Issue.id.desc()).all()


@router.post("", response_model=schemas.IssueOut)
def create_issue(payload: schemas.IssueIn, db: Session = Depends(get_db)):
    issue = models.Issue(**payload.model_dump())
    db.add(issue)
    db.commit()
    db.refresh(issue)
    issue.key = _generate_key(db, issue.project_id, issue.id)
    db.add(models.ActivityLogEntry(issue_id=issue.id, actor="System", field="created", new_value=issue.title))
    db.commit()
    db.refresh(issue)
    return issue


@router.get("/{issue_id}", response_model=schemas.IssueOut)
def get_issue(issue_id: int, db: Session = Depends(get_db)):
    issue = db.query(models.Issue).get(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")
    return issue


@router.patch("/{issue_id}", response_model=schemas.IssueOut)
def update_issue(issue_id: int, payload: schemas.IssueUpdate, db: Session = Depends(get_db)):
    issue = db.query(models.Issue).get(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")

    data = payload.model_dump(exclude_unset=True)
    label_ids = data.pop("label_ids", None)
    component_ids = data.pop("component_ids", None)

    for k, v in data.items():
        old = getattr(issue, k)
        if k in TRACKED_FIELDS:
            _log_activity(db, issue.id, k, old, v)
        setattr(issue, k, v)

    if label_ids is not None:
        issue.labels = db.query(models.Label).filter(models.Label.id.in_(label_ids)).all()
    if component_ids is not None:
        issue.components = db.query(models.Component).filter(models.Component.id.in_(component_ids)).all()

    db.commit()
    db.refresh(issue)
    return issue


@router.delete("/{issue_id}")
def delete_issue(issue_id: int, db: Session = Depends(get_db)):
    issue = db.query(models.Issue).get(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")
    db.delete(issue)
    db.commit()
    return {"deleted": True}


# ---------- Comments ----------

@router.get("/{issue_id}/comments", response_model=list[schemas.CommentOut])
def list_comments(issue_id: int, db: Session = Depends(get_db)):
    return db.query(models.Comment).filter(models.Comment.issue_id == issue_id).order_by(models.Comment.created_at).all()


@router.post("/{issue_id}/comments", response_model=schemas.CommentOut)
def add_comment(issue_id: int, payload: schemas.CommentIn, db: Session = Depends(get_db)):
    if not db.query(models.Issue).get(issue_id):
        raise HTTPException(404, "Issue not found")
    comment = models.Comment(issue_id=issue_id, author_name=payload.author_name, body=payload.body)
    db.add(comment)
    db.add(models.ActivityLogEntry(issue_id=issue_id, actor=payload.author_name, field="comment", new_value=payload.body[:80]))
    db.commit()
    db.refresh(comment)
    return comment


@router.delete("/comments/{comment_id}")
def delete_comment(comment_id: int, db: Session = Depends(get_db)):
    comment = db.query(models.Comment).get(comment_id)
    if not comment:
        raise HTTPException(404, "Comment not found")
    db.delete(comment)
    db.commit()
    return {"deleted": True}


# ---------- Watchers ----------

@router.get("/{issue_id}/watchers", response_model=list[schemas.WatcherOut])
def list_watchers(issue_id: int, db: Session = Depends(get_db)):
    rows = db.query(models.Watcher).filter(models.Watcher.issue_id == issue_id).all()
    out = []
    for w in rows:
        emp = db.query(models.Employee).get(w.employee_id)
        out.append(schemas.WatcherOut(id=w.id, issue_id=w.issue_id, employee_id=w.employee_id, employee_name=emp.name if emp else None))
    return out


@router.post("/{issue_id}/watchers", response_model=schemas.WatcherOut)
def add_watcher(issue_id: int, payload: schemas.WatcherIn, db: Session = Depends(get_db)):
    existing = db.query(models.Watcher).filter(
        models.Watcher.issue_id == issue_id, models.Watcher.employee_id == payload.employee_id
    ).first()
    if existing:
        raise HTTPException(400, "Already watching")
    w = models.Watcher(issue_id=issue_id, employee_id=payload.employee_id)
    db.add(w)
    db.commit()
    db.refresh(w)
    emp = db.query(models.Employee).get(payload.employee_id)
    return schemas.WatcherOut(id=w.id, issue_id=w.issue_id, employee_id=w.employee_id, employee_name=emp.name if emp else None)


@router.delete("/{issue_id}/watchers/{employee_id}")
def remove_watcher(issue_id: int, employee_id: int, db: Session = Depends(get_db)):
    w = db.query(models.Watcher).filter(models.Watcher.issue_id == issue_id, models.Watcher.employee_id == employee_id).first()
    if not w:
        raise HTTPException(404, "Not watching")
    db.delete(w)
    db.commit()
    return {"deleted": True}


# ---------- Issue Links ----------

@router.get("/{issue_id}/links", response_model=list[schemas.IssueLinkOut])
def list_links(issue_id: int, db: Session = Depends(get_db)):
    rows = db.query(models.IssueLink).filter(models.IssueLink.source_issue_id == issue_id).all()
    out = []
    for l in rows:
        target = db.query(models.Issue).get(l.target_issue_id)
        out.append(schemas.IssueLinkOut(
            id=l.id, source_issue_id=l.source_issue_id, target_issue_id=l.target_issue_id,
            link_type=l.link_type, target_title=target.title if target else None, target_key=target.key if target else None,
        ))
    return out


@router.post("/{issue_id}/links", response_model=schemas.IssueLinkOut)
def add_link(issue_id: int, payload: schemas.IssueLinkIn, db: Session = Depends(get_db)):
    if not db.query(models.Issue).get(payload.target_issue_id):
        raise HTTPException(404, "Target issue not found")
    link = models.IssueLink(source_issue_id=issue_id, target_issue_id=payload.target_issue_id, link_type=payload.link_type)
    db.add(link)
    db.commit()
    db.refresh(link)
    target = db.query(models.Issue).get(payload.target_issue_id)
    return schemas.IssueLinkOut(
        id=link.id, source_issue_id=link.source_issue_id, target_issue_id=link.target_issue_id,
        link_type=link.link_type, target_title=target.title, target_key=target.key,
    )


@router.delete("/links/{link_id}")
def delete_link(link_id: int, db: Session = Depends(get_db)):
    link = db.query(models.IssueLink).get(link_id)
    if not link:
        raise HTTPException(404, "Link not found")
    db.delete(link)
    db.commit()
    return {"deleted": True}


# ---------- Activity ----------

@router.get("/{issue_id}/activity", response_model=list[schemas.ActivityLogOut])
def list_activity(issue_id: int, db: Session = Depends(get_db)):
    return db.query(models.ActivityLogEntry).filter(models.ActivityLogEntry.issue_id == issue_id).order_by(models.ActivityLogEntry.created_at.desc()).all()
