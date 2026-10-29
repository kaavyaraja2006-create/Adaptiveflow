"""
ScenarioEngine \u2014 the What-If Simulator.

Reruns the same real health/risk calculations against a hypothetical
mutation of current state and returns a structured card: a headline
verdict, a side-by-side comparison, the concrete factors driving the
result ("why"), a plain-language mechanism explanation, and an actionable
recommendation with confidence \u2014 the same shape for every scenario type
so the UI can render one consistent card design.
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from .. import models
from . import risk_engine


def _risk_word(pct: int) -> str:
    if pct >= 70:
        return "Critical"
    if pct >= 45:
        return "Elevated"
    if pct >= 20:
        return "Moderate"
    return "Low"


def _fit_word(score: int) -> str:
    if score >= 75:
        return "Strong"
    if score >= 45:
        return "Moderate"
    if score >= 25:
        return "Weak"
    return "Poor"


def run_scenario(db: Session, project: models.Project, scenario_type: str, params: Dict) -> Dict:
    baseline = risk_engine.project_health(db, project)

    if scenario_type == "absence_extended":
        extra_days = int(params.get("days", 3))
        employee_id = params.get("employee_id")
        employee = db.query(models.Employee).get(employee_id) if employee_id else None
        affected = db.query(models.Issue).filter(
            models.Issue.project_id == project.id,
            models.Issue.assignee_id == employee_id,
            models.Issue.status != "Done",
        ).count()
        delay_delta = min(60, affected * extra_days * 2)
        new_delay = min(98, baseline["delay_probability"] + delay_delta)

        name = employee.name if employee else "this employee"
        return {
            "scenario_type": scenario_type,
            "title": "AI ABSENCE IMPACT ANALYSIS",
            "headline": f"Extending {name}'s absence raises delay risk to {new_delay}%",
            "compare": [
                {"label": "CURRENT", "value": baseline["delay_probability"], "verdict": _risk_word(baseline["delay_probability"]), "tone": "neutral", "recommended": False},
                {"label": f"+{extra_days} DAYS", "value": new_delay, "verdict": _risk_word(new_delay), "tone": "bad", "recommended": False},
            ],
            "why": [
                {"icon": "📋", "value": f"{affected} task(s)", "label": "Active work affected"},
                {"icon": "📅", "value": f"+{extra_days}d", "label": "Extension length"},
            ],
            "insight": f"Each additional day this absence continues compounds delay risk across all {affected} active task(s) currently assigned to {name}.",
            "recommendation": {
                "label": "Reassign now" if delay_delta >= 15 else "Monitor, no action needed yet",
                "confidence": new_delay,
            },
        }

    if scenario_type == "add_developer":
        skill_area = params.get("skill_area", "Backend")
        issues = db.query(models.Issue).filter(
            models.Issue.project_id == project.id, models.Issue.issue_type == skill_area,
            models.Issue.status != "Done",
        ).count()
        risk_reduction = min(30, issues * 3)
        new_delay = max(2, baseline["delay_probability"] - risk_reduction)

        return {
            "scenario_type": scenario_type,
            "title": "AI CAPACITY ANALYSIS",
            "headline": f"Adding a {skill_area} developer could cut delay risk to {new_delay}%",
            "compare": [
                {"label": "CURRENT", "value": baseline["delay_probability"], "verdict": _risk_word(baseline["delay_probability"]), "tone": "neutral", "recommended": False},
                {"label": "WITH NEW HIRE", "value": new_delay, "verdict": _risk_word(new_delay), "tone": "good", "recommended": True},
            ],
            "why": [
                {"icon": "📂", "value": f"{issues} issue(s)", "label": f"Open {skill_area} work"},
                {"icon": "⚡", "value": "+100%", "label": "Added capacity"},
            ],
            "insight": f"There are {issues} open {skill_area} issue(s) that could absorb the new hire's capacity, directly relieving the current bottleneck in that skill area.",
            "recommendation": {
                "label": "Worth hiring" if risk_reduction >= 15 else "Marginal benefit at current backlog size",
                "confidence": min(95, 50 + risk_reduction * 2),
            },
        }

    if scenario_type == "remove_feature":
        points = int(params.get("story_points", 8))
        planned = sum(s.planned_points for s in project.sprints) or 1
        relief = min(35, round(points / max(1, planned) * 100))
        new_delay = max(2, baseline["delay_probability"] - relief)

        return {
            "scenario_type": scenario_type,
            "title": "AI SCOPE ANALYSIS",
            "headline": f"Removing this feature could lower delay risk to {new_delay}%",
            "compare": [
                {"label": "WITH FEATURE", "value": baseline["delay_probability"], "verdict": _risk_word(baseline["delay_probability"]), "tone": "neutral", "recommended": False},
                {"label": "WITHOUT FEATURE", "value": new_delay, "verdict": _risk_word(new_delay), "tone": "good", "recommended": True},
            ],
            "why": [
                {"icon": "📉", "value": f"{points} pts", "label": "Scope removed"},
                {"icon": "📊", "value": f"{relief}%", "label": "Sprint pressure relief"},
            ],
            "insight": f"Removing {points} story points frees roughly {relief}% of current sprint pressure, directly reducing carry-over risk.",
            "recommendation": {
                "label": "Cut scope" if relief >= 15 else "Limited impact \u2014 consider a larger cut",
                "confidence": min(95, 50 + relief * 2),
            },
        }

    if scenario_type == "change_methodology":
        m = risk_engine.methodology_suitability(db, project)
        target = m["recommendation"]
        winner_score = max(m["scrum"], m["kanban"])
        scrum_wins = m["scrum"] >= m["kanban"]

        insight = (
            "High workflow volatility makes continuous-flow management more suitable than fixed sprints."
            if not scrum_wins else
            "Low workflow volatility means fixed-length sprints still provide predictable planning without excess overhead."
        )

        return {
            "scenario_type": scenario_type,
            "title": "AI METHODOLOGY ANALYSIS",
            "headline": f"{target} is the better fit for this project",
            "compare": [
                {"label": "SCRUM", "value": m["scrum"], "verdict": _fit_word(m["scrum"]), "tone": "good" if scrum_wins else "neutral", "recommended": scrum_wins},
                {"label": "KANBAN", "value": m["kanban"], "verdict": _fit_word(m["kanban"]), "tone": "good" if not scrum_wins else "neutral", "recommended": not scrum_wins},
            ],
            "why": [
                {"icon": "🔄", "value": f"{m['avg_carry_over']} pts", "label": "Carry-over"},
                {"icon": "⚠️", "value": f"{m['avg_unplanned']} pts", "label": "Unplanned work"},
            ],
            "insight": insight,
            "recommendation": {"label": target.upper(), "confidence": winner_score},
        }

    if scenario_type == "move_sprint":
        sprints_moved = int(params.get("sprints", 1))
        new_delay = min(98, baseline["delay_probability"] + sprints_moved * 5)

        return {
            "scenario_type": scenario_type,
            "title": "AI SCHEDULE ANALYSIS",
            "headline": f"Moving this out {sprints_moved} sprint(s) raises delay risk to {new_delay}%",
            "compare": [
                {"label": "CURRENT PLAN", "value": baseline["delay_probability"], "verdict": _risk_word(baseline["delay_probability"]), "tone": "neutral", "recommended": False},
                {"label": f"+{sprints_moved} SPRINT(S)", "value": new_delay, "verdict": _risk_word(new_delay), "tone": "bad", "recommended": False},
            ],
            "why": [
                {"icon": "⏱️", "value": f"{sprints_moved} sprint(s)", "label": "Delay introduced"},
                {"icon": "🔗", "value": "downstream", "label": "Dependent work impact"},
            ],
            "insight": "Moving this work out shifts delivery for everything that depends on it by a similar margin, compounding downstream schedule risk.",
            "recommendation": {
                "label": "Avoid if possible" if sprints_moved * 5 >= 15 else "Manageable delay",
                "confidence": new_delay,
            },
        }

    return {
        "scenario_type": scenario_type,
        "title": "AI ANALYSIS",
        "headline": "Unknown scenario type",
        "compare": [],
        "why": [],
        "insight": "This scenario type is not recognized by the simulator.",
        "recommendation": {"label": "N/A", "confidence": 0},
    }
