"""
Phase 4-7 endpoints: the AdaptiveFlow Project Digital Twin — dependency
graph, uncertainty-aware forecasting, and Monte Carlo what-if simulation.
Additive to the existing /api/simulate flow (still works unchanged).
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .. import models
from ..database import get_db
from ..ai import digital_twin, graph_engine, forecasting_engine

router = APIRouter(prefix="/api/simulator", tags=["digital-twin"])


class ScenarioIn(BaseModel):
    type: str
    employee_id: Optional[int] = None
    days: Optional[int] = None
    issue_id: Optional[int] = None
    delta_points: Optional[int] = None
    block_days: Optional[int] = None
    reduction_pct: Optional[float] = None
    capacity_hours_per_day: Optional[float] = None
    daily_hours: Optional[float] = None


class RunRequest(BaseModel):
    project_id: int
    sprint_id: Optional[int] = None
    iterations: int = 1000


class WhatIfRequest(BaseModel):
    project_id: int
    sprint_id: Optional[int] = None
    scenario: ScenarioIn
    iterations: int = 500


def _get_project(db: Session, project_id: int) -> models.Project:
    project = db.query(models.Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@router.post("/run")
def run_baseline(payload: RunRequest, db: Session = Depends(get_db)):
    project = _get_project(db, payload.project_id)
    result = digital_twin.analyze(db, project, sprint_id=payload.sprint_id, iterations=payload.iterations)
    result["triggers"] = digital_twin.check_triggers(result)
    return result


@router.post("/whatif")
def run_whatif(payload: WhatIfRequest, db: Session = Depends(get_db)):
    project = _get_project(db, payload.project_id)
    baseline = digital_twin.analyze(db, project, sprint_id=payload.sprint_id, iterations=payload.iterations)
    scenario_dict = payload.scenario.model_dump(exclude_unset=True)
    scenario_result = digital_twin.analyze(
        db, project, sprint_id=payload.sprint_id, scenario=scenario_dict, iterations=payload.iterations
    )
    return {
        "baseline": baseline,
        "scenario": scenario_result,
        "delta": {
            "sprint_risk": round(scenario_result["sprint_risk"] - baseline["sprint_risk"], 3),
            "sprint_miss_probability": round(
                scenario_result["components"]["p_miss"] - baseline["components"]["p_miss"], 3
            ),
            "expected_delay_hours": round(
                scenario_result["simulation"]["expected_delay_hours"] - baseline["simulation"]["expected_delay_hours"], 1
            ),
        },
        "triggers": digital_twin.check_triggers(scenario_result),
    }


@router.get("/graph/{project_id}")
def get_graph(project_id: int, db: Session = Depends(get_db)):
    project = _get_project(db, project_id)
    g = graph_engine.build_project_graph(db, project)
    dep_graph = graph_engine.task_dependency_subgraph(g)

    duration_lookup = {}
    for node, data in g.nodes(data=True):
        if data.get("kind") == "Task":
            f = forecasting_engine.predict(data.get("story_points") or 3, data.get("issue_type") or "Backend")
            duration_lookup[node] = f["expected_remaining_hours"]

    cp = graph_engine.critical_path(g, duration_lookup)
    return {
        "summary": graph_engine.graph_summary(g),
        "critical_path": cp,
        "nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, "type": d.get("edge_type")} for u, v, d in g.edges(data=True)],
    }


@router.get("/history/{project_id}")
def simulation_history(project_id: int, db: Session = Depends(get_db)):
    runs = (
        db.query(models.SimulationRun)
        .filter(models.SimulationRun.project_id == project_id)
        .order_by(models.SimulationRun.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {"id": r.id, "scenario_type": r.scenario_type, "created_at": r.created_at,
         "sprint_risk": r.results_json.get("sprint_risk"), "sprint_risk_band": r.results_json.get("sprint_risk_band"),
         "sprint_miss_probability": r.results_json.get("components", {}).get("p_miss")}
        for r in runs
    ]


@router.get("/models/duration/status")
def duration_model_status():
    return forecasting_engine.get_status()


@router.post("/models/duration/train")
def train_duration_model(db: Session = Depends(get_db)):
    return forecasting_engine.train(db)
