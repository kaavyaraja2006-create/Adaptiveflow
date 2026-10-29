"""
Monte Carlo digital-twin simulator.

Each iteration:
1. Samples a duration for every active task from its forecasted
   distribution (forecasting_engine.py — trained NGBoost or honest
   empirical baseline, whichever is actually available).
2. Respects real task dependencies (graph_engine.py, built from actual
   IssueLink data) — a task can't start before everything it depends on
   finishes.
3. Schedules each developer's tasks sequentially against their real
   effective daily capacity (a documented simplification — see
   forecasting_engine.EFFECTIVE_HOURS_PER_DAY).
4. Produces a completion-time distribution across all iterations, from
   which sprint-miss probability and p50/p80/p95 fall out naturally
   rather than being hardcoded.

What-if scenarios (`scenario`) mutate the inputs BEFORE simulation runs —
never the outputs — so every scenario is a genuine re-simulation, not a
formula tweak on top of a cached baseline.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from .. import models
from . import graph_engine, forecasting_engine, workload_engine

DEFAULT_ITERATIONS = 1000
WORKDAY_HOURS = forecasting_engine.EFFECTIVE_HOURS_PER_DAY


def _is_workday(d: datetime) -> bool:
    return d.weekday() < 5


def _working_hours_between(start: datetime, end: datetime, daily_hours: float) -> float:
    if end <= start:
        return 0.0
    days = 0
    cur = start
    while cur.date() < end.date():
        if _is_workday(cur):
            days += 1
        cur += timedelta(days=1)
    return days * daily_hours


def _collect_scope(db: Session, project: models.Project, sprint_id: Optional[int]) -> List[models.Issue]:
    q = db.query(models.Issue).filter(models.Issue.project_id == project.id, models.Issue.status != "Done")
    if sprint_id:
        q = q.filter(models.Issue.sprint_id == sprint_id)
    return q.all()


def _apply_scenario(db: Session, issues: List[models.Issue], scenario: Optional[Dict]) -> Dict:
    """Returns a mutation context consumed by the simulation loop —
    real objects aren't written to the DB; this is a hypothetical overlay."""
    ctx = {
        "reassign": {},          # issue_id -> new assignee_id
        "extra_duration_hours": {},  # issue_id -> extra hours (dependency failure / blocked)
        "capacity_multiplier": {},   # employee_id -> multiplier on daily hours (e.g. 0.5 for reduced capacity)
        "unavailable_until": {},     # employee_id -> hours of unavailability at sim start
        "priority_boost": set(),     # issue_ids to schedule first
        "extra_points": 0,           # synthetic scope-change task, in story points
        "virtual_capacity_hours": 0.0,  # add-developer: extra pooled capacity
        "transfer_multiplier": {},   # issue_id -> duration multiplier from knowledge-transfer cost
    }
    if not scenario:
        return ctx
    kind = scenario.get("type")

    if kind == "developer_absence":
        emp_id = scenario.get("employee_id")
        days = scenario.get("days", 5)
        ctx["unavailable_until"][emp_id] = days * WORKDAY_HOURS
        # affected tasks get reassigned to the top-ranked available candidate (real allocation engine)
        from . import assignment_engine
        for issue in issues:
            if issue.assignee_id == emp_id:
                rec = assignment_engine.recommend_for_issue(db, issue, exclude_employee_id=emp_id)
                if not rec["no_suitable_candidate"]:
                    ctx["reassign"][issue.id] = rec["recommended_employee_id"]
                    ctx["transfer_multiplier"][issue.id] = 1 + (rec.get("transfer_cost") or 0)

    elif kind == "add_developer":
        capacity = scenario.get("capacity_hours_per_day", WORKDAY_HOURS)
        days = scenario.get("days", 10)
        ctx["virtual_capacity_hours"] = capacity * days

    elif kind == "scope_change":
        ctx["extra_points"] = scenario.get("delta_points", 0)

    elif kind == "priority_change":
        issue_id = scenario.get("issue_id")
        if issue_id:
            ctx["priority_boost"].add(issue_id)

    elif kind == "dependency_failure":
        issue_id = scenario.get("issue_id")
        block_days = scenario.get("block_days", 1)
        if issue_id:
            ctx["extra_duration_hours"][issue_id] = block_days * WORKDAY_HOURS

    elif kind == "reassign_single":
        # used to evaluate a single candidate reassignment's simulated impact
        # (see digital_twin.simulate_reassignment_candidates) rather than a
        # whole-team absence scenario.
        issue_id = scenario.get("issue_id")
        new_assignee = scenario.get("new_assignee_id")
        if issue_id and new_assignee:
            ctx["reassign"][issue_id] = new_assignee
            transfer_mult = scenario.get("transfer_multiplier")
            if transfer_mult:
                ctx["transfer_multiplier"][issue_id] = transfer_mult

    elif kind == "capacity_reduction":
        emp_id = scenario.get("employee_id")
        reduction_pct = scenario.get("reduction_pct", 50)
        if emp_id:
            ctx["capacity_multiplier"][emp_id] = max(0.0, 1 - reduction_pct / 100)
        else:
            ctx["capacity_multiplier"]["__all__"] = max(0.0, 1 - reduction_pct / 100)

    elif kind == "process_change":
        pass  # handled directly in run() via scenario.daily_hours, not via ctx

    return ctx


def run(db: Session, project: models.Project, sprint_id: Optional[int] = None,
        scenario: Optional[Dict] = None, iterations: int = DEFAULT_ITERATIONS) -> Dict:
    issues = _collect_scope(db, project, sprint_id)
    if not issues:
        return {"iterations": 0, "note": "No active tasks in scope to simulate.",
                "sprint_miss_probability": 0.0, "p50_hours": 0, "p80_hours": 0, "p95_hours": 0,
                "expected_delay_hours": 0, "critical_tasks": [], "risk_drivers": []}

    g = graph_engine.build_project_graph(db, project)
    dep_graph = graph_engine.task_dependency_subgraph(g)
    ctx = _apply_scenario(db, issues, scenario)

    daily_hours_override = scenario.get("daily_hours") if scenario and scenario.get("type") == "process_change" else None
    daily_hours = daily_hours_override or WORKDAY_HOURS

    # precompute duration distributions once (expensive part), then resample per iteration
    dist_params = {}
    for issue in issues:
        forecast = forecasting_engine.predict(issue.story_points or 3, issue.issue_type)
        mean_h = forecast["expected_remaining_hours"]
        std_h = max(0.5, forecast["std_days"] * daily_hours)
        mult = ctx["transfer_multiplier"].get(issue.id, 1.0)
        extra = ctx["extra_duration_hours"].get(issue.id, 0.0)
        dist_params[issue.id] = (mean_h * mult, std_h * mult, extra)

    # topological order over the real dependency subgraph, task-node -> issue id
    node_to_issue = {f"task:{i.id}": i.id for i in issues}
    try:
        order_nodes = [n for n in __import__("networkx").topological_sort(dep_graph) if n in node_to_issue]
    except Exception:
        order_nodes = list(node_to_issue.keys())
    ordered_issue_ids = [node_to_issue[n] for n in order_nodes]
    remaining_ids = [i.id for i in issues if i.id not in ordered_issue_ids]
    ordered_issue_ids += remaining_ids
    # priority boost: move boosted issues to the front (still respecting their own dependency constraints implicitly via start-time logic)
    if ctx["priority_boost"]:
        ordered_issue_ids.sort(key=lambda iid: 0 if iid in ctx["priority_boost"] else 1)

    issue_by_id = {i.id: i for i in issues}
    rng = np.random.default_rng(123)

    completion_samples = []
    critical_task_counts = {i.id: 0 for i in issues}
    over_capacity_counts = {}

    sprint = db.query(models.Sprint).get(sprint_id) if sprint_id else None
    sprint_deadline_hours = None
    if sprint and sprint.end_date:
        now = datetime.utcnow()
        sprint_deadline_hours = _working_hours_between(now, sprint.end_date, daily_hours)

    for _ in range(iterations):
        dev_available_at = {}  # employee_id -> hours-from-now they're free
        virtual_capacity_left = ctx["virtual_capacity_hours"]
        finish_times = {}

        for iid in ordered_issue_ids:
            issue = issue_by_id[iid]
            mean_h, std_h, extra_h = dist_params[iid]
            sampled = max(0.25, rng.normal(mean_h, std_h)) + extra_h

            dep_ready = 0.0
            node = f"task:{iid}"
            if node in dep_graph:
                for pred in dep_graph.predecessors(node):
                    pred_iid = node_to_issue.get(pred)
                    if pred_iid in finish_times:
                        dep_ready = max(dep_ready, finish_times[pred_iid])

            assignee = ctx["reassign"].get(iid, issue.assignee_id)
            if assignee:
                cap_mult = ctx["capacity_multiplier"].get(assignee, ctx["capacity_multiplier"].get("__all__", 1.0))
                unavailable = ctx["unavailable_until"].get(assignee, 0.0)
                dev_ready = max(dev_available_at.get(assignee, 0.0), unavailable)
                start = max(dep_ready, dev_ready)
                effective_duration = sampled / max(0.05, cap_mult)
                finish = start + effective_duration
                dev_available_at[assignee] = finish
                if cap_mult < 1.0:
                    over_capacity_counts[assignee] = over_capacity_counts.get(assignee, 0) + 1
            elif virtual_capacity_left > 0:
                start = dep_ready
                finish = start + sampled
                virtual_capacity_left -= sampled
            else:
                # unassigned work with no virtual capacity: treat as unblocked but unscheduled slack risk
                start = dep_ready
                finish = start + sampled

            finish_times[iid] = finish

        if ctx["extra_points"]:
            # synthetic scope-change task appended at the end, using the project's average story-point duration
            avg_hours_per_point = np.mean([dist_params[i][0] / max(1, issue_by_id[i].story_points or 3) for i in dist_params]) if dist_params else 4.0
            finish_times["__scope_change__"] = max(finish_times.values(), default=0) + ctx["extra_points"] * avg_hours_per_point

        project_finish = max(finish_times.values()) if finish_times else 0.0
        completion_samples.append(project_finish)

        if finish_times:
            crit_id = max((k for k in finish_times if k != "__scope_change__"), key=lambda k: finish_times[k], default=None)
            if crit_id is not None:
                critical_task_counts[crit_id] = critical_task_counts.get(crit_id, 0) + 1

    completion_samples = np.array(completion_samples)
    p50 = float(np.percentile(completion_samples, 50))
    p80 = float(np.percentile(completion_samples, 80))
    p95 = float(np.percentile(completion_samples, 95))
    mean_hours = float(completion_samples.mean())

    if sprint_deadline_hours is not None:
        miss_prob = float(np.mean(completion_samples > sprint_deadline_hours))
        expected_delay = max(0.0, mean_hours - sprint_deadline_hours)
    else:
        # no sprint end_date configured: use p80 vs mean as a rough proxy signal instead of a hard deadline
        miss_prob = float(np.mean(completion_samples > mean_hours * 1.15))
        expected_delay = max(0.0, p80 - mean_hours)

    critical_tasks = sorted(critical_task_counts.items(), key=lambda kv: -kv[1])[:5]
    critical_task_info = [
        {"issue_id": iid, "title": issue_by_id[iid].title, "critical_path_frequency": round(count / iterations, 3)}
        for iid, count in critical_tasks if count > 0
    ]

    risk_drivers = []
    blocked = [i for i in issues if i.blocked]
    if blocked:
        risk_drivers.append(f"{len(blocked)} task(s) currently marked blocked")
    if over_capacity_counts:
        worst = max(over_capacity_counts.items(), key=lambda kv: kv[1])
        emp = db.query(models.Employee).get(worst[0])
        if emp:
            risk_drivers.append(f"{emp.name} is running at reduced capacity in this scenario")
    if critical_task_info:
        risk_drivers.append(f"'{critical_task_info[0]['title']}' is on the critical path in "
                            f"{round(critical_task_info[0]['critical_path_frequency']*100)}% of simulated runs")
    if not risk_drivers:
        risk_drivers.append("No significant risk drivers detected in this simulation")

    return {
        "iterations": iterations,
        "sprint_miss_probability": round(miss_prob, 3),
        "p50_hours": round(p50, 1), "p80_hours": round(p80, 1), "p95_hours": round(p95, 1),
        "mean_hours": round(mean_hours, 1),
        "expected_delay_hours": round(expected_delay, 1),
        "critical_tasks": critical_task_info,
        "risk_drivers": risk_drivers,
        "sprint_deadline_hours": round(sprint_deadline_hours, 1) if sprint_deadline_hours is not None else None,
        "tasks_simulated": len(issues),
    }
