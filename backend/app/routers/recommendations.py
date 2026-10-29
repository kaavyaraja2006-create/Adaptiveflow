from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.get("/recommendations", response_model=list[schemas.RecommendationOut])
def list_recommendations(status: str | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Recommendation).order_by(models.Recommendation.created_at.desc())
    if status:
        q = q.filter(models.Recommendation.status == status)
    return q.all()


@router.post("/recommendations/{rec_id}/decide", response_model=schemas.RecommendationOut)
def decide(rec_id: int, payload: schemas.RecommendationDecision, db: Session = Depends(get_db)):
    rec = db.query(models.Recommendation).get(rec_id)
    if not rec:
        raise HTTPException(404, "Recommendation not found")
    rec.status = payload.decision
    db.add(models.DecisionHistoryEntry(
        recommendation=rec.title,
        model="AdaptiveFlow Engine v1",
        confidence=rec.confidence,
        evidence=rec.evidence,
        human_decision=payload.decision.capitalize(),
        outcome="Pending outcome" if payload.decision == "approved" else "Not executed",
    ))
    db.commit()
    db.refresh(rec)
    return rec


@router.get("/decisions", response_model=list[schemas.DecisionHistoryOut])
def list_decisions(db: Session = Depends(get_db)):
    return db.query(models.DecisionHistoryEntry).order_by(models.DecisionHistoryEntry.created_at.desc()).all()


@router.get("/decisions/performance")
def model_performance(db: Session = Depends(get_db)):
    entries = db.query(models.DecisionHistoryEntry).all()
    total = len(entries) or 1
    approved = len([e for e in entries if e.human_decision == "Approved"])
    avg_confidence = round(sum(e.confidence for e in entries) / total) if entries else 0
    return {
        "total_decisions": len(entries),
        "acceptance_rate": round(approved / total * 100),
        "average_confidence": avg_confidence,
        "by_model": _group_by_model(entries),
    }


def _group_by_model(entries):
    groups: dict = {}
    for e in entries:
        groups.setdefault(e.model, {"count": 0, "avg_confidence": 0, "confidences": []})
        groups[e.model]["count"] += 1
        groups[e.model]["confidences"].append(e.confidence)
    for g in groups.values():
        g["avg_confidence"] = round(sum(g["confidences"]) / len(g["confidences"])) if g["confidences"] else 0
        del g["confidences"]
    return groups
