"""
EmployeeCapabilityEngine.

Every other engine (assignment, absence recovery, sprint planning) reads
from here rather than recomputing its own notion of "how good is this
person at this kind of work" \u2014 one shared, explainable source of truth
derived from WorkHistoryItem rows, not a hardcoded number.
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from .. import models


def success_rate_for(history: List[models.WorkHistoryItem], task_type: str) -> Dict:
    matching = [h for h in history if h.task_type == task_type]
    if not matching:
        return {"rate": 50, "n": 0, "on_time_rate": 50}
    successful = [h for h in matching if h.success]
    on_time = [h for h in matching if h.on_time]
    return {
        "rate": round(len(successful) / len(matching) * 100),
        "n": len(matching),
        "on_time_rate": round(len(on_time) / len(matching) * 100),
    }


def capability_profile(employee: models.Employee) -> Dict:
    history = employee.history
    task_types = sorted({h.task_type for h in history} | set((employee.skills or {}).keys()))
    breakdown = {}
    for t in task_types:
        sr = success_rate_for(history, t)
        breakdown[t] = {
            "skill": int((employee.skills or {}).get(t, 50)),
            "success_rate": sr["rate"],
            "on_time_rate": sr["on_time_rate"],
            "completed_tasks": sr["n"],
        }
    avg_success = round(sum(b["success_rate"] for b in breakdown.values()) / len(breakdown)) if breakdown else 50
    return {
        "employee_id": employee.id,
        "name": employee.name,
        "role": employee.role,
        "status": employee.status,
        "utilization": round(employee.utilization),
        "available_capacity": round(max(0.0, 100 - employee.utilization)),
        "avg_success_rate": avg_success,
        "total_completed": len(history),
        "breakdown": breakdown,
    }


def team_workload(db: Session, team_id: int = None) -> List[Dict]:
    q = db.query(models.Employee)
    if team_id:
        q = q.filter(models.Employee.team_id == team_id)
    out = []
    for e in q.all():
        out.append({
            "employee_id": e.id,
            "name": e.name,
            "team_id": e.team_id,
            "status": e.status,
            "utilization": round(e.utilization),
            "available_capacity": round(max(0.0, 100 - e.utilization)),
            "overloaded": e.utilization >= 85,
            "underutilized": e.utilization <= 30,
        })
    return out
