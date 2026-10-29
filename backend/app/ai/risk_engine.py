"""
RiskEngine + DelayPredictionEngine + Project Health scoring.

Computed from real signals on the project's issues and sprints: how much
work is blocked, how overloaded the assigned team is, how much carried
over from the last sprint, and how much of the work is unplanned. Every
score states which signals drove it.
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from .. import models
from . import capability_engine


def project_health(db: Session, project: models.Project) -> Dict:
    issues = db.query(models.Issue).filter(models.Issue.project_id == project.id).all()
    sprints = db.query(models.Sprint).filter(models.Sprint.project_id == project.id).all()

    total = len(issues) or 1
    done = len([i for i in issues if i.status == "Done"])
    blocked = len([i for i in issues if i.blocked])
    progress_pct = round(done / total * 100)

    assignee_ids = {i.assignee_id for i in issues if i.assignee_id}
    employees = db.query(models.Employee).filter(models.Employee.id.in_(assignee_ids)).all() if assignee_ids else []
    overloaded = [e for e in employees if e.utilization >= 85]

    latest_sprint = max(sprints, key=lambda s: s.id) if sprints else None
    carry_over = latest_sprint.carry_over_points if latest_sprint else 0
    unplanned = latest_sprint.unplanned_points if latest_sprint else 0
    planned = latest_sprint.planned_points if latest_sprint else 1

    schedule_health = max(10, 100 - blocked * 10 - carry_over * 2)
    resource_health = max(10, 100 - len(overloaded) * 20)
    risk_signals = blocked * 12 + len(overloaded) * 15 + carry_over * 3 + unplanned * 4
    risk_health = max(5, 100 - risk_signals)

    health_score = round(schedule_health * 0.35 + resource_health * 0.3 + risk_health * 0.35)

    delay_probability = min(95, max(5, round(risk_signals * 1.3)))
    predicted_delay_days = round(delay_probability / 100 * 12)

    reasons = []
    if blocked:
        reasons.append(f"{blocked} issue(s) currently blocked")
    if overloaded:
        reasons.append(f"{len(overloaded)} team member(s) at or above 85% utilization")
    if carry_over:
        reasons.append(f"{carry_over} story points carried over from the last sprint")
    if unplanned:
        reasons.append(f"{unplanned} unplanned story points absorbed this sprint")
    if not reasons:
        reasons.append("No significant risk signals detected in current data")

    return {
        "project_id": project.id,
        "project_name": project.name,
        "health_score": health_score,
        "schedule_health": schedule_health,
        "risk_health": risk_health,
        "resource_health": resource_health,
        "progress_pct": progress_pct,
        "delay_probability": delay_probability,
        "predicted_delay_days": predicted_delay_days,
        "reasons": reasons,
    }


def methodology_suitability(db: Session, project: models.Project) -> Dict:
    sprints = db.query(models.Sprint).filter(models.Sprint.project_id == project.id).all()
    if not sprints:
        return {"scrum": 70, "kanban": 50, "recommendation": "Scrum", "confidence": 40,
                "avg_carry_over": 0.0, "avg_unplanned": 0.0,
                "reasons": ["Not enough sprint history yet to evaluate methodology fit."]}

    avg_carry_over = sum(s.carry_over_points for s in sprints) / len(sprints)
    avg_unplanned = sum(s.unplanned_points for s in sprints) / len(sprints)
    avg_planned = sum(s.planned_points for s in sprints) / max(1, len(sprints))

    volatility = (avg_carry_over + avg_unplanned) / max(1, avg_planned)
    scrum_score = max(10, round(90 - volatility * 100))
    kanban_score = min(95, round(40 + volatility * 100))

    reasons = [
        f"average sprint carry-over is {avg_carry_over:.1f} points",
        f"average unplanned work per sprint is {avg_unplanned:.1f} points",
    ]
    recommendation = "Kanban" if kanban_score > scrum_score else "Scrum"
    return {
        "scrum": scrum_score, "kanban": kanban_score,
        "recommendation": recommendation,
        "confidence": min(90, 50 + len(sprints) * 5),
        "avg_carry_over": round(avg_carry_over, 1),
        "avg_unplanned": round(avg_unplanned, 1),
        "reasons": reasons,
    }


def bottlenecks(db: Session, project: models.Project) -> List[str]:
    issues = db.query(models.Issue).filter(models.Issue.project_id == project.id).all()
    by_type: Dict[str, List[models.Issue]] = {}
    for i in issues:
        by_type.setdefault(i.issue_type, []).append(i)

    findings = []
    for t, group in by_type.items():
        blocked = [i for i in group if i.blocked]
        in_progress = [i for i in group if i.status == "In Progress"]
        if len(blocked) >= 2:
            findings.append(f"{t} has {len(blocked)} blocked issues \u2014 likely a dependency bottleneck.")
        if len(in_progress) >= 4:
            findings.append(f"{t} has {len(in_progress)} issues in progress simultaneously \u2014 possible capacity bottleneck.")
    if not findings:
        findings.append("No clear bottleneck detected from current issue data.")
    return findings
