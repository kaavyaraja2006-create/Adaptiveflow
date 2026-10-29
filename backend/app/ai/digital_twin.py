"""
AdaptiveFlow Project Digital Twin — orchestrator.

Ties together the data-quality gate, dependency graph, duration
forecasting, and Monte Carlo simulation into the single composite
SprintRisk score the spec defines:

    SprintRisk = 0.35*P_miss + 0.20*CriticalDependencyRisk
               + 0.15*WorkloadRisk + 0.15*BlockedTaskRisk + 0.15*Uncertainty

Every component is computed from the real simulation/graph/workload
outputs above — none of the five inputs is a placeholder.
"""
from typing import Dict, Optional
from sqlalchemy.orm import Session
from .. import models
from . import graph_engine, monte_carlo_engine, workload_engine, data_quality, audit_ledger


def _risk_band(score: float) -> str:
    if score >= 0.80:
        return "Critical"
    if score >= 0.60:
        return "High"
    if score >= 0.35:
        return "Moderate"
    return "Low"


def analyze(db: Session, project: models.Project, sprint_id: Optional[int] = None,
            scenario: Optional[Dict] = None, iterations: int = 1000) -> Dict:
    audit_ledger.record_event(
        db, event_type="SIMULATION_STARTED",
        payload={"project_id": project.id, "sprint_id": sprint_id, "scenario_type": (scenario or {}).get("type", "baseline"),
                 "iterations": iterations},
        actor_id="system-ai", project_id=project.id,
    )

    dq = data_quality.run_project_data_quality_scan(db)

    g = graph_engine.build_project_graph(db, project)
    dep_graph = graph_engine.task_dependency_subgraph(g)
    graph_summary = graph_engine.graph_summary(g)

    sim = monte_carlo_engine.run(db, project, sprint_id=sprint_id, scenario=scenario, iterations=iterations)

    issues = db.query(models.Issue).filter(models.Issue.project_id == project.id, models.Issue.status != "Done")
    if sprint_id:
        issues = issues.filter(models.Issue.sprint_id == sprint_id)
    issues = issues.all()
    total = len(issues) or 1
    blocked = [i for i in issues if i.blocked]
    blocked_task_risk = min(1.0, len(blocked) / total)

    # critical-dependency risk: what share of the tasks on the simulated
    # critical path are ALSO blocked or have unresolved dependencies
    critical_ids = {c["issue_id"] for c in sim.get("critical_tasks", [])}
    critical_blocked = sum(1 for i in issues if i.id in critical_ids and i.blocked)
    critical_dependency_risk = (critical_blocked / len(critical_ids)) if critical_ids else 0.0

    assignee_ids = {i.assignee_id for i in issues if i.assignee_id}
    workload_risks = []
    for eid in assignee_ids:
        emp = db.query(models.Employee).get(eid)
        if emp and not emp.workload_opt_out:
            wp = workload_engine.workload_profile(db, emp)
            workload_risks.append(wp.recovery_risk)
    workload_risk = (sum(workload_risks) / len(workload_risks)) if workload_risks else 0.0

    uncertainty_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
    from . import forecasting_engine
    uncertainties = []
    for i in issues:
        f = forecasting_engine.predict(i.story_points or 3, i.issue_type)
        uncertainties.append(uncertainty_map.get(f["uncertainty"], 0.5))
    uncertainty_score = (sum(uncertainties) / len(uncertainties)) if uncertainties else 0.5

    p_miss = sim.get("sprint_miss_probability", 0.0)
    sprint_risk = (
        0.35 * p_miss + 0.20 * critical_dependency_risk + 0.15 * workload_risk
        + 0.15 * blocked_task_risk + 0.15 * uncertainty_score
    )
    sprint_risk = round(max(0.0, min(1.0, sprint_risk)), 3)

    result = {
        "project_id": project.id,
        "project_name": project.name,
        "sprint_id": sprint_id,
        "sprint_risk": sprint_risk,
        "sprint_risk_band": _risk_band(sprint_risk),
        "components": {
            "p_miss": round(p_miss, 3),
            "critical_dependency_risk": round(critical_dependency_risk, 3),
            "workload_risk": round(workload_risk, 3),
            "blocked_task_risk": round(blocked_task_risk, 3),
            "uncertainty": round(uncertainty_score, 3),
        },
        "simulation": sim,
        "graph_summary": graph_summary,
        "data_quality": dq,
        "scenario": scenario,
    }

    run = models.SimulationRun(
        project_id=project.id,
        scenario_type=(scenario or {}).get("type", "baseline"),
        scenario_json=scenario or {},
        assumptions_json={"iterations": iterations, "sprint_id": sprint_id},
        results_json=result,
    )
    db.add(run)
    db.commit()
    result["simulation_run_id"] = run.id

    audit_ledger.record_event(
        db, event_type="SIMULATION_COMPLETED",
        payload={"project_id": project.id, "sprint_id": sprint_id, "simulation_run_id": run.id,
                 "sprint_risk": sprint_risk, "sprint_risk_band": _risk_band(sprint_risk),
                 "sprint_miss_probability": p_miss},
        actor_id="system-ai", project_id=project.id, model_version="digital-twin-v1.0",
    )
    return result


TRIGGERS = [
    dict(code="delay_risk", condition=lambda r: r["components"]["p_miss"] > 0.60,
         message="Sprint-miss probability rose above 60% — reallocation analysis recommended."),
    dict(code="overload_risk", condition=lambda r: r["components"]["workload_risk"] > 0.75,
         message="Average team recovery risk exceeds 0.75 — avoid assigning new high-effort tasks."),
    dict(code="dependency_blockage", condition=lambda r: r["components"]["blocked_task_risk"] > 0.25,
         message="More than a quarter of active tasks are blocked — downstream impact simulation recommended."),
    dict(code="low_confidence", condition=lambda r: r["components"]["uncertainty"] > 0.65,
         message="Forecast uncertainty is high — recommend manager review rather than automated action."),
]


def check_triggers(result: Dict) -> list:
    fired = []
    for t in TRIGGERS:
        try:
            if t["condition"](result):
                fired.append({"code": t["code"], "message": t["message"]})
        except Exception:
            continue
    return fired


def simulate_reassignment_candidates(db: Session, issue: models.Issue, candidates: list, iterations: int = 150) -> list:
    """The key connection between the allocation engine and the digital
    twin: for each candidate, actually re-run the Monte Carlo simulation
    with them assigned instead of running scoring formulas alone, and
    report each candidate's simulated sprint-success probability alongside
    their allocation score — e.g. "Candidate A: 81% simulated sprint
    success vs Candidate B: 74%" — real simulated outcomes, not a
    restatement of the allocation score in different units."""
    project = db.query(models.Project).get(issue.project_id)
    if not project or not candidates:
        return []

    results = []
    for c in candidates[:3]:
        scenario = {
            "type": "reassign_single",
            "issue_id": issue.id,
            "new_assignee_id": c["employee_id"],
            "transfer_multiplier": 1 + (c.get("transfer_cost") or 0),
        }
        sim = monte_carlo_engine.run(db, project, sprint_id=issue.sprint_id, scenario=scenario, iterations=iterations)
        results.append({
            "employee_id": c["employee_id"],
            "name": c["name"],
            "allocation_score": c.get("allocation_score") or c.get("overall"),
            "simulated_sprint_success_probability": round(1 - sim["sprint_miss_probability"], 3),
            "simulated_expected_delay_hours": sim["expected_delay_hours"],
        })
    results.sort(key=lambda r: -r["simulated_sprint_success_probability"])
    return results
