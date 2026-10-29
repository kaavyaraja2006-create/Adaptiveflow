from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..ai import risk_engine, scenario_engine

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard/health", response_model=list[schemas.ProjectHealthOut])
def dashboard_health(db: Session = Depends(get_db)):
    projects = db.query(models.Project).all()
    return [risk_engine.project_health(db, p) for p in projects]


@router.get("/dashboard/methodology/{project_id}")
def methodology(project_id: int, db: Session = Depends(get_db)):
    project = db.query(models.Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return risk_engine.methodology_suitability(db, project)


@router.get("/dashboard/bottlenecks/{project_id}")
def bottlenecks(project_id: int, db: Session = Depends(get_db)):
    project = db.query(models.Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return {"findings": risk_engine.bottlenecks(db, project)}


@router.post("/simulate", response_model=schemas.ScenarioResult)
def simulate(payload: schemas.ScenarioRequest, db: Session = Depends(get_db)):
    project = db.query(models.Project).get(payload.project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return scenario_engine.run_scenario(db, project, payload.scenario_type, payload.params)
