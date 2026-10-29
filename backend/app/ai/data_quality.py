"""
Data Quality Gate.

Every AI pipeline in AdaptiveFlow (allocation, workload/recovery risk, and
later the digital twin) must pass through this gate before it's allowed to
treat data as trustworthy input. Bad data is FLAGGED or EXCLUDED — never
silently treated as valid.

This module also enforces the project's target-leakage rule: prediction
features may only use information available BEFORE prediction time. None
of the functions here read post-hoc fields (final resolution date, actual
final effort, final sprint outcome) as inputs to a "live" prediction —
those fields only ever appear in WorkHistoryItem rows for COMPLETED work,
which is legitimate historical training/reference signal, not a leak.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session
from .. import models

STALE_DAYS_THRESHOLD = 21          # no activity log entry / status change in this many days
MIN_STORY_POINTS = 1
MAX_STORY_POINTS = 21               # generous upper bound; higher signals a data-entry error
STALE_SNAPSHOT_DAYS = 3             # workload snapshot older than this is "stale" for recovery-risk purposes


@dataclass
class DataQualityReport:
    reliable: bool
    flags: List[str] = field(default_factory=list)

    def add(self, flag: str):
        self.flags.append(flag)
        self.reliable = False


def check_issue(db: Session, issue: models.Issue) -> DataQualityReport:
    """Data-quality check for a single task before it's used as allocation
    input. Returns a report the allocation engine must respect — a task
    that fails this check should be flagged in the UI and, where the issue
    is severe (invalid points, cancelled/duplicate), excluded from
    automatic scoring rather than silently scored anyway."""
    report = DataQualityReport(reliable=True)

    if issue.story_points is None or issue.story_points < MIN_STORY_POINTS:
        report.add(f"Invalid story points ({issue.story_points}) — task excluded from automated scoring")
    elif issue.story_points > MAX_STORY_POINTS:
        report.add(f"Story points ({issue.story_points}) unusually high — flagged for manager review")

    if issue.status == "Done":
        report.add("Issue is already Done — not eligible for reallocation")

    if getattr(issue, "item_type", None) and issue.item_type not in (
        "Epic-adjacent", "Story", "Task", "Bug", "Sub-task", "Improvement", "Spike"
    ):
        report.add(f"Unrecognized item_type '{issue.item_type}'")

    # staleness: no activity log entry and no status change signal in a long time
    # while still nominally "in progress" is a sign the ticket isn't being
    # actively tracked and its status shouldn't be trusted at face value.
    if issue.status == "In Progress":
        last_activity = (
            db.query(models.ActivityLogEntry)
            .filter(models.ActivityLogEntry.issue_id == issue.id)
            .order_by(models.ActivityLogEntry.created_at.desc())
            .first()
        )
        reference_time = last_activity.created_at if last_activity else issue.created_at
        if reference_time and (datetime.utcnow() - reference_time) > timedelta(days=STALE_DAYS_THRESHOLD):
            report.add(
                f"No recorded activity in {STALE_DAYS_THRESHOLD}+ days while marked In Progress — "
                "status may be stale"
            )

    return report


def check_employee_snapshot_freshness(latest_snapshot_date) -> List[str]:
    """Workload/recovery-risk data is only as trustworthy as how recently
    it was computed. Flag (don't silently use) stale snapshots."""
    if latest_snapshot_date is None:
        return ["No workload activity data available for this employee yet"]
    if (datetime.utcnow() - latest_snapshot_date) > timedelta(days=STALE_SNAPSHOT_DAYS):
        return [f"Workload data is more than {STALE_SNAPSHOT_DAYS} days old — recovery risk may be outdated"]
    return []


def run_project_data_quality_scan(db: Session) -> dict:
    """Broader sweep used by dashboards / the digital twin to report overall
    data health rather than gate a single decision."""
    issues = db.query(models.Issue).all()
    flags = {"stale_tickets": 0, "invalid_story_points": 0, "duplicate_active_assignments": 0,
              "missing_skills_employees": 0, "unassigned_teams": 0}

    for issue in issues:
        report = check_issue(db, issue)
        for f in report.flags:
            if "stale" in f.lower():
                flags["stale_tickets"] += 1
            if "invalid story points" in f.lower():
                flags["invalid_story_points"] += 1

    # duplicate active assignment: same employee with more issues "In Progress"
    # than is plausible to be genuinely working simultaneously
    from collections import Counter
    active_by_employee = Counter(
        i.assignee_id for i in issues if i.status == "In Progress" and i.assignee_id
    )
    flags["duplicate_active_assignments"] = sum(1 for v in active_by_employee.values() if v > 5)

    employees = db.query(models.Employee).all()
    flags["missing_skills_employees"] = sum(1 for e in employees if not e.skills)
    flags["unassigned_teams"] = sum(1 for e in employees if not e.team_id)

    total_checks = max(1, len(issues) + len(employees))
    total_flagged = sum(flags.values())
    data_health_pct = max(0, round(100 - (total_flagged / total_checks) * 100))

    return {"data_health_pct": data_health_pct, "flags": flags, "checked_issues": len(issues), "checked_employees": len(employees)}
