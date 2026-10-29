from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..ai import capability_engine

router = APIRouter(prefix="/api", tags=["teams"])


@router.get("/teams", response_model=list[schemas.TeamOut])
def list_teams(db: Session = Depends(get_db)):
    return db.query(models.Team).all()


@router.post("/teams", response_model=schemas.TeamOut)
def create_team(payload: schemas.TeamIn, db: Session = Depends(get_db)):
    team = models.Team(name=payload.name, description=payload.description)
    db.add(team)
    db.commit()
    db.refresh(team)
    return team


@router.patch("/teams/{team_id}", response_model=schemas.TeamOut)
def update_team(team_id: int, payload: schemas.TeamIn, db: Session = Depends(get_db)):
    team = db.query(models.Team).get(team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    team.name = payload.name
    team.description = payload.description
    db.commit()
    db.refresh(team)
    return team


@router.delete("/teams/{team_id}")
def delete_team(team_id: int, db: Session = Depends(get_db)):
    team = db.query(models.Team).get(team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    db.delete(team)
    db.commit()
    return {"deleted": True}


@router.get("/employees", response_model=list[schemas.EmployeeOut])
def list_employees(team_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Employee)
    if team_id:
        q = q.filter(models.Employee.team_id == team_id)
    return q.all()


@router.post("/employees", response_model=schemas.EmployeeOut)
def create_employee(payload: schemas.EmployeeIn, db: Session = Depends(get_db)):
    emp = models.Employee(name=payload.name, role=payload.role, team_id=payload.team_id, skills=payload.skills, office_id=payload.office_id)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return emp


@router.patch("/employees/{employee_id}", response_model=schemas.EmployeeOut)
def update_employee(employee_id: int, payload: schemas.EmployeeUpdate, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(emp, k, v)
    db.commit()
    db.refresh(emp)
    return emp


@router.delete("/employees/{employee_id}")
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    db.delete(emp)
    db.commit()
    return {"deleted": True}


@router.get("/employees/{employee_id}/profile")
def employee_profile(employee_id: int, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    return capability_engine.capability_profile(emp)


@router.get("/workload")
def workload(team_id: int | None = None, db: Session = Depends(get_db)):
    return capability_engine.team_workload(db, team_id)


@router.get("/projects", response_model=list[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).all()


@router.get("/sprints", response_model=list[schemas.SprintOut])
def list_sprints(project_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Sprint)
    if project_id:
        q = q.filter(models.Sprint.project_id == project_id)
    return q.all()
