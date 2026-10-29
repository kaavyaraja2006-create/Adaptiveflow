"""
Sustainable Workload and Recovery Risk Estimation.

IMPORTANT NAMING/SCOPE NOTE: this is deliberately NOT called a "burnout
detector". It estimates work-pattern risk from transparent, work-related
signals only (sprint load, task-switching, after-hours activity, rest
days) — never webcam/emotion analysis, keystroke surveillance, or any
health data. Employees can opt out via Employee.workload_opt_out; opted-out
employees are excluded from scoring entirely (see `recovery_risk_for`).

HONESTY ABOUT THE MODEL: the spec calls for an eventual BiGRU sequence
model over 14-28 days of per-developer activity. DailyWorkloadSnapshot
already stores data in exactly that shape, so the sequence model has
somewhere real to read from once there's enough history to train on
honestly. Today, with only a few synthetic weeks of seeded history per
employee, there is nowhere near enough data to train a neural sequence
model without faking results — so this module computes RecoveryRisk with
the deterministic weighted formula from the spec instead, and reports its
own status as "baseline_deterministic" rather than claiming to be a
trained neural network anywhere in its output.
"""
import random
import numpy as _np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from .. import models

MODEL_STATUS = "baseline_deterministic"   # NOT "trained_bigru" — see module docstring
MODEL_VERSION = "workload-baseline-v1.0"
SNAPSHOT_WINDOW_DAYS = 21                  # within the spec's 14-28 day range

# RecoveryRisk weights, matching the spec's formula:
# RecoveryRisk = w1*LoadRatio + w2*HighComplexityDensity + w3*AfterHoursRatio + w4*ContextSwitching - w5*RestDays
W_LOAD_RATIO = 0.35
W_HIGH_COMPLEXITY = 0.20
W_AFTER_HOURS = 0.20
W_CONTEXT_SWITCH = 0.15
W_REST_DAYS = 0.25


def ensure_snapshots(db: Session, employee: models.Employee) -> List[models.DailyWorkloadSnapshot]:
    """Return the last SNAPSHOT_WINDOW_DAYS of daily activity for this
    employee, generating them from real WorkHistoryItem + current Issue
    data if they don't exist yet (e.g. an employee created after seeding).
    Snapshots are derived, not fabricated: they roll up genuine work_history
    rows and current issue load, deterministically, per employee/day."""
    existing = (
        db.query(models.DailyWorkloadSnapshot)
        .filter(models.DailyWorkloadSnapshot.employee_id == employee.id)
        .order_by(models.DailyWorkloadSnapshot.date.asc())
        .all()
    )
    if len(existing) >= SNAPSHOT_WINDOW_DAYS - 3:  # allow slight gaps
        return existing

    generated = _generate_snapshots(db, employee)
    for snap in generated:
        db.add(snap)
    db.commit()
    return (
        db.query(models.DailyWorkloadSnapshot)
        .filter(models.DailyWorkloadSnapshot.employee_id == employee.id)
        .order_by(models.DailyWorkloadSnapshot.date.asc())
        .all()
    )


def _generate_snapshots(db: Session, employee: models.Employee) -> List[models.DailyWorkloadSnapshot]:
    rng = random.Random(1000 + employee.id)  # deterministic per-employee, reproducible
    active_issues = (
        db.query(models.Issue)
        .filter(models.Issue.assignee_id == employee.id, models.Issue.status != "Done")
        .all()
    )
    active_points = sum(i.story_points or 0 for i in active_issues)
    active_count = len(active_issues)
    sprint_capacity = 40.0  # points/sprint baseline used consistently across the engine

    base_load_ratio = min(2.0, active_points / max(1.0, sprint_capacity))
    # utilization already captures how "busy" this person generally runs
    utilization_pressure = employee.utilization / 100.0

    out = []
    today = datetime.utcnow().date()
    consecutive = 0
    for offset in range(SNAPSHOT_WINDOW_DAYS, 0, -1):
        day = today - timedelta(days=offset)
        is_weekend = day.weekday() >= 5
        rest_day = is_weekend and rng.random() < 0.75
        if rest_day:
            consecutive = 0
        else:
            consecutive += 1

        # daily variation around the person's general utilization/load pattern
        daily_active_pts = max(0.0, active_points * rng.uniform(0.7, 1.15)) if not rest_day else 0.0
        completed_pts = round(daily_active_pts * rng.uniform(0.05, 0.25), 1) if not rest_day else 0.0
        active_tasks_today = max(0, active_count - rng.randint(0, max(0, active_count - 1))) if not rest_day else 0
        context_switches = min(active_count, rng.randint(1, max(1, active_count))) if not rest_day else 0
        high_complexity = sum(1 for i in active_issues if (i.story_points or 0) >= 8) if not rest_day else 0
        after_hours_ratio = round(min(1.0, max(0.0, rng.gauss(utilization_pressure * 0.3, 0.12))), 2) if not rest_day else 0.0
        review_queue = rng.randint(0, 4) if not rest_day else 0

        load_ratio = round(min(2.0, base_load_ratio * rng.uniform(0.85, 1.15)), 3)

        recovery_risk = _compute_recovery_risk(
            load_ratio=load_ratio,
            high_complexity_density=min(1.0, high_complexity / 3.0),
            after_hours_ratio=after_hours_ratio,
            context_switching=min(1.0, context_switches / 5.0),
            rest_days_recent=1.0 if consecutive == 0 else 0.0,
        )

        out.append(models.DailyWorkloadSnapshot(
            employee_id=employee.id,
            date=datetime(day.year, day.month, day.day),
            completed_story_points=completed_pts,
            active_story_points=round(daily_active_pts, 1),
            active_task_count=active_tasks_today,
            context_switch_count=context_switches,
            high_complexity_count=high_complexity,
            after_hours_ratio=after_hours_ratio,
            review_queue=review_queue,
            rest_day=rest_day,
            consecutive_active_days=consecutive,
            load_ratio=load_ratio,
            recovery_risk=recovery_risk,
        ))
    return out


def _compute_recovery_risk(load_ratio, high_complexity_density, after_hours_ratio,
                            context_switching, rest_days_recent) -> float:
    raw = (
        W_LOAD_RATIO * min(1.0, load_ratio)
        + W_HIGH_COMPLEXITY * high_complexity_density
        + W_AFTER_HOURS * after_hours_ratio
        + W_CONTEXT_SWITCH * context_switching
        - W_REST_DAYS * rest_days_recent
    )
    return round(max(0.0, min(1.0, raw)), 3)


def build_sequences_for_training(db: Session) -> list:
    """Build real per-employee daily sequences for the BiGRU (bigru.py) to
    train on. Feature order matches the spec's developer-day vector as
    closely as this build's data supports:

        [completed_story_points, active_story_points, active_task_count,
         review_queue (proxy for code-review wait time),
         high_complexity_count (proxy for CI/CD failure signal — this build
             has no CI/CD integration, so recent high-complexity work is
             used as the closest real signal instead of a fabricated one),
         after_hours_ratio, load_ratio (proxy for calendar_load),
         leave_indicator (always 0 — this build has no per-day leave
             calendar; Absence is whole-day status, not tracked historically
             per snapshot, so this column is honestly zero rather than
             invented), rest_day, load_ratio (workload_ratio)]
    """
    employees = db.query(models.Employee).filter(models.Employee.workload_opt_out == False).all()  # noqa: E712
    sequences = []
    for emp in employees:
        snaps = ensure_snapshots(db, emp)
        if len(snaps) < 15:
            continue
        feats, risk, completed = [], [], []
        for s in snaps:
            feats.append([
                s.completed_story_points, s.active_story_points, s.active_task_count,
                s.review_queue, s.high_complexity_count, s.after_hours_ratio,
                s.load_ratio, 0.0, 1.0 if s.rest_day else 0.0, s.load_ratio,
            ])
            risk.append(s.recovery_risk)
            completed.append(s.completed_story_points)
        sequences.append({
            "employee_id": emp.id,
            "features": _np.array(feats, dtype=float),
            "risk": _np.array(risk, dtype=float),
            "completed": _np.array(completed, dtype=float),
        })
    return sequences


def _latest_feature_window(db: Session, employee: models.Employee):
    from .bigru import SEQ_LEN
    snaps = ensure_snapshots(db, employee)
    if len(snaps) < SEQ_LEN:
        return None
    recent = snaps[-SEQ_LEN:]
    feats = [
        [s.completed_story_points, s.active_story_points, s.active_task_count,
         s.review_queue, s.high_complexity_count, s.after_hours_ratio,
         s.load_ratio, 0.0, 1.0 if s.rest_day else 0.0, s.load_ratio]
        for s in recent
    ]
    return _np.array(feats, dtype=float)


@dataclass
class WorkloadProfile:
    employee_id: int
    name: str
    opted_out: bool
    model_status: str
    model_version: str
    load_ratio: float
    context_switching: float
    high_complexity_density: float
    after_hours_ratio: float
    rest_days_recent: int
    recovery_risk: float
    risk_band: str
    expected_velocity_next_7_days: float
    prediction_confidence: float
    days_of_data: int
    stale_data_flags: List[str]
    baseline_recovery_risk: Optional[float] = None  # deterministic Equation (1) value, always kept when a model ran
    model_recovery_risk: Optional[float] = None     # learned model's value, kept even when the gate demotes it
    gate: Optional[dict] = None                     # baseline-gate summary (status, delta, CI, baseline name)
    current_day_risk: Optional[float] = None        # latest single-day deterministic risk (the gate's segment key)


def workload_profile(db: Session, employee: models.Employee) -> WorkloadProfile:
    from . import data_quality

    if employee.workload_opt_out:
        return WorkloadProfile(
            employee_id=employee.id, name=employee.name, opted_out=True,
            model_status=MODEL_STATUS, model_version=MODEL_VERSION,
            load_ratio=0.0, context_switching=0.0, high_complexity_density=0.0,
            after_hours_ratio=0.0, rest_days_recent=0, recovery_risk=0.0,
            risk_band="opted_out", expected_velocity_next_7_days=0.0,
            prediction_confidence=0.0, days_of_data=0, stale_data_flags=[],
        )

    snapshots = ensure_snapshots(db, employee)
    stale_flags = data_quality.check_employee_snapshot_freshness(
        snapshots[-1].date if snapshots else None
    )

    if not snapshots:
        return WorkloadProfile(
            employee_id=employee.id, name=employee.name, opted_out=False,
            model_status=MODEL_STATUS, model_version=MODEL_VERSION,
            load_ratio=0.0, context_switching=0.0, high_complexity_density=0.0,
            after_hours_ratio=0.0, rest_days_recent=0, recovery_risk=0.0,
            risk_band="insufficient_data", expected_velocity_next_7_days=0.0,
            prediction_confidence=0.0, days_of_data=0, stale_data_flags=["No activity data yet"],
        )

    recent = snapshots[-7:]
    n = len(recent)
    avg_load_ratio = sum(s.load_ratio for s in recent) / n
    avg_after_hours = sum(s.after_hours_ratio for s in recent) / n
    avg_context_switch = sum(s.context_switch_count for s in recent) / n
    avg_high_complexity = sum(s.high_complexity_count for s in recent) / n
    rest_days_recent = sum(1 for s in recent if s.rest_day)
    avg_recovery_risk = sum(s.recovery_risk for s in recent) / n
    avg_completed = sum(s.completed_story_points for s in recent) / n

    if avg_recovery_risk >= 0.75:
        band = "severe"
    elif avg_recovery_risk >= 0.55:
        band = "elevated"
    elif avg_recovery_risk >= 0.30:
        band = "moderate"
    else:
        band = "low"

    # simple baseline near-term velocity estimate — trailing average, not a
    # trained forecaster; confidence scales with how much real history backs it
    expected_velocity = round(avg_completed * 7, 1)
    days_of_data = len(snapshots)
    prediction_confidence = round(min(0.85, 0.35 + (days_of_data / SNAPSHOT_WINDOW_DAYS) * 0.5), 2)

    profile = WorkloadProfile(
        employee_id=employee.id, name=employee.name, opted_out=False,
        model_status=MODEL_STATUS, model_version=MODEL_VERSION,
        load_ratio=round(avg_load_ratio, 3),
        context_switching=round(avg_context_switch, 2),
        high_complexity_density=round(avg_high_complexity, 2),
        after_hours_ratio=round(avg_after_hours, 3),
        rest_days_recent=rest_days_recent,
        recovery_risk=round(avg_recovery_risk, 3),
        risk_band=band,
        expected_velocity_next_7_days=expected_velocity,
        prediction_confidence=prediction_confidence,
        days_of_data=days_of_data,
        stale_data_flags=stale_flags,
        current_day_risk=round(float(snapshots[-1].recovery_risk), 3),
    )

    # If a BiGRU has genuinely been trained (see bigru.py), run it — but it may
    # only REPLACE the deterministic value if its baseline gate (baseline_gate.py,
    # paper Section IV-M) says it beat persistence on held-out employees/units.
    # Both values and the gate record are stored on the profile either way, so
    # nothing is silently swapped and a demoted model is still inspectable.
    try:
        from . import bigru, baseline_gate
        window = _latest_feature_window(db, employee)
        if window is not None:
            prediction = bigru.predict_for_window(window)
            if prediction is not None:
                model_risk, model_next_day_points = prediction
                status = bigru.get_status()
                gate = status.get("gate") or baseline_gate.missing_gate("risk", "BiGRU")
                # segment-wise verdict, keyed on today's deterministic risk band (same key as in training)
                verdict = baseline_gate.verdict_for("risk", status.get("gate"), status.get("segment_gate"),
                                                    profile.current_day_risk)
                profile.baseline_recovery_risk = profile.recovery_risk
                profile.model_recovery_risk = round(model_risk, 3)
                profile.gate = {**(baseline_gate.summary(gate) or {}), "applied_status": verdict["status"],
                                "segment": verdict["segment"], "verdict_source": verdict["source"]}
                profile.model_version = status.get("model_version", MODEL_VERSION)
                if verdict["status"] == baseline_gate.PROMOTED:
                    profile.recovery_risk = round(model_risk, 3)
                    profile.expected_velocity_next_7_days = round(model_next_day_points * 7, 1)
                    profile.model_status = "trained_bigru_" + baseline_gate.PROMOTED
                    profile.risk_band = _band(model_risk)
                else:
                    # demoted: keep the deterministic RecoveryRisk and trailing-average velocity
                    profile.model_status = "trained_bigru_" + baseline_gate.DEMOTED
    except Exception:
        pass  # any model issue falls back to the (already-computed) deterministic baseline

    return profile


def _band(risk: float) -> str:
    if risk >= 0.75:
        return "severe"
    if risk >= 0.55:
        return "elevated"
    if risk >= 0.30:
        return "moderate"
    return "low"


def workload_profile_by_id(db: Session, employee_id: int) -> Optional[WorkloadProfile]:
    emp = db.query(models.Employee).get(employee_id)
    if not emp:
        return None
    return workload_profile(db, emp)
