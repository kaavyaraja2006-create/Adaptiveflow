from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api", tags=["labels-components"])


@router.get("/labels", response_model=list[schemas.LabelOut])
def list_labels(db: Session = Depends(get_db)):
    return db.query(models.Label).all()


@router.post("/labels", response_model=schemas.LabelOut)
def create_label(payload: schemas.LabelIn, db: Session = Depends(get_db)):
    existing = db.query(models.Label).filter(models.Label.name == payload.name).first()
    if existing:
        return existing
    label = models.Label(name=payload.name, color=payload.color)
    db.add(label)
    db.commit()
    db.refresh(label)
    return label


@router.delete("/labels/{label_id}")
def delete_label(label_id: int, db: Session = Depends(get_db)):
    label = db.query(models.Label).get(label_id)
    if not label:
        raise HTTPException(404, "Label not found")
    db.delete(label)
    db.commit()
    return {"deleted": True}


@router.get("/components", response_model=list[schemas.ComponentOut])
def list_components(project_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Component)
    if project_id:
        q = q.filter(models.Component.project_id == project_id)
    return q.all()


@router.post("/components", response_model=schemas.ComponentOut)
def create_component(payload: schemas.ComponentIn, db: Session = Depends(get_db)):
    comp = models.Component(name=payload.name, project_id=payload.project_id)
    db.add(comp)
    db.commit()
    db.refresh(comp)
    return comp


@router.delete("/components/{component_id}")
def delete_component(component_id: int, db: Session = Depends(get_db)):
    comp = db.query(models.Component).get(component_id)
    if not comp:
        raise HTTPException(404, "Component not found")
    db.delete(comp)
    db.commit()
    return {"deleted": True}
