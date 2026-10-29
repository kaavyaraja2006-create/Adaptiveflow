from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..ai import idea_engine

router = APIRouter(prefix="/api/ideas", tags=["ideas"])


@router.get("", response_model=list[schemas.IdeaOut])
def list_ideas(project_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Idea).order_by(models.Idea.created_at.desc())
    if project_id:
        q = q.filter(models.Idea.project_id == project_id)
    return q.all()


@router.post("/analyze", response_model=schemas.IdeaAnalysisOut)
def analyze(payload: schemas.IdeaIn, db: Session = Depends(get_db)):
    """Run the AI understanding pipeline WITHOUT persisting \u2014 lets the UI
    show the analysis and let the user Accept/Edit/Reject before saving."""
    text = f"{payload.title}. {payload.description}".strip()
    result = idea_engine.analyze_idea(db, text)
    return result


@router.post("", response_model=schemas.IdeaOut)
def create_idea(payload: schemas.IdeaIn, db: Session = Depends(get_db)):
    text = f"{payload.title}. {payload.description}".strip()
    analysis = idea_engine.analyze_idea(db, text)
    idea = models.Idea(
        title=payload.title,
        description=payload.description,
        problem=analysis["problem"],
        project_id=payload.project_id,
        team=analysis["team"],
        goal=analysis["goal"],
        impact=analysis["impact"],
        effort=analysis["effort"],
        confidence=analysis["confidence"],
        priority=analysis["priority"],
        priority_reason=analysis["priority_reason"],
    )
    db.add(idea)
    db.commit()
    db.refresh(idea)

    if analysis["duplicate"]:
        rec = models.Recommendation(
            kind="duplicate", title=f"Possible duplicate of \u201c{analysis['duplicate']['title']}\u201d",
            reason=f"{analysis['duplicate']['pct']}% similarity to an existing idea.",
            evidence=f"Existing idea #{analysis['duplicate']['idea_id']}",
            confidence=analysis["duplicate"]["pct"], related_idea_id=idea.id,
        )
        db.add(rec)
        db.commit()
    return idea


@router.get("/{idea_id}", response_model=schemas.IdeaOut)
def get_idea(idea_id: int, db: Session = Depends(get_db)):
    idea = db.query(models.Idea).get(idea_id)
    if not idea:
        raise HTTPException(404, "Idea not found")
    return idea


@router.patch("/{idea_id}", response_model=schemas.IdeaOut)
def update_idea(idea_id: int, payload: schemas.IdeaUpdate, db: Session = Depends(get_db)):
    idea = db.query(models.Idea).get(idea_id)
    if not idea:
        raise HTTPException(404, "Idea not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(idea, k, v)
    db.commit()
    db.refresh(idea)
    return idea


@router.delete("/{idea_id}")
def delete_idea(idea_id: int, db: Session = Depends(get_db)):
    idea = db.query(models.Idea).get(idea_id)
    if not idea:
        raise HTTPException(404, "Idea not found")
    db.delete(idea)
    db.commit()
    return {"deleted": True}


@router.get("/{idea_id}/similar")
def similar_ideas(idea_id: int, db: Session = Depends(get_db)):
    idea = db.query(models.Idea).get(idea_id)
    if not idea:
        raise HTTPException(404, "Idea not found")
    text = f"{idea.title}. {idea.description}"
    return idea_engine.find_similar(db, text, exclude_id=idea_id)
