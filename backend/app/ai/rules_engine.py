"""
Symbolic Rule Engine (R1-R10).

Transparent, manager-editable rules that sit ABOVE the scoring model.
A candidate excluded by a hard rule can never become eligible again just
because their allocation score is high — hard exclusions are absolute.

Rules are stored in the AllocationRule table so a manager can disable them
or retune thresholds/weights later without a code change. `ensure_default_rules`
seeds the standard R1-R10 set the first time the app runs against a fresh
database (safe to call repeatedly — it only inserts rows that don't exist).
"""
from dataclasses import dataclass, field
from typing import List, Optional
from sqlalchemy.orm import Session
from .. import models

DEFAULT_RULES = [
    dict(code="R1", name="Minimum skill confidence", kind="exclusion", threshold=70.0,
         description="Candidate's skill match for this task type must be at least the threshold, "
                      "or they're excluded outright."),
    dict(code="R2", name="Leave / unavailable exclusion", kind="exclusion", threshold=None,
         description="Candidates on approved leave or already marked absent are excluded."),
    dict(code="R3", name="Workload ceiling", kind="exclusion", threshold=1.0,
         description="Candidates with a load ratio above the threshold are excluded, unless emergency "
                      "mode is enabled."),
    dict(code="R4", name="Severe recovery risk", kind="penalty", threshold=0.75, weight=30.0,
         description="Recovery risk above the threshold applies a severe score penalty (or exclusion "
                      "at the very top of the range)."),
    dict(code="R5", name="Security-critical ownership", kind="review", threshold=None,
         description="Security-critical tasks assigned to someone without prior ownership in that "
                      "area require explicit manager review before approval."),
    dict(code="R6", name="Recent high-complexity load", kind="penalty", threshold=2.0, weight=10.0,
         description="Two or more high-complexity tasks touched in the last 48 hours adds a workload "
                      "penalty to the candidate's score."),
    dict(code="R7", name="Low-confidence top match", kind="review", threshold=70.0,
         description="If the top-ranked candidate has the best skill match but confidence below the "
                      "threshold, recommend pairing them with a reviewer/mentor."),
    dict(code="R8", name="No eligible candidate", kind="escalation", threshold=None,
         description="When no candidate survives the rule filters, escalate to a contingency workflow "
                      "(split task, de-scope, extend deadline, request external support)."),
    dict(code="R9", name="Fairness ceiling", kind="penalty", threshold=1.5, weight=15.0,
         description="An allocation that would push a candidate's workload deviation from the team "
                      "median beyond the threshold is penalized to protect team-wide fairness."),
    dict(code="R10", name="Insufficient confidence", kind="info", threshold=0.60,
         description="Recommendations with confidence below the threshold are shown as \"Insufficient "
                      "Confidence\" rather than automated — a manager must decide manually."),
]


def ensure_default_rules(db: Session) -> None:
    existing_codes = {r.code for r in db.query(models.AllocationRule).all()}
    for rule in DEFAULT_RULES:
        if rule["code"] in existing_codes:
            continue
        db.add(models.AllocationRule(
            code=rule["code"], name=rule["name"], description=rule["description"],
            kind=rule["kind"], enabled=True,
            threshold=rule.get("threshold"), weight=rule.get("weight"),
        ))
    db.commit()


def get_rules(db: Session) -> dict:
    ensure_default_rules(db)
    return {r.code: r for r in db.query(models.AllocationRule).all()}


def rules_snapshot(db: Session) -> list:
    """Plain-data copy of the rule set in force, for decision receipts."""
    return [{"code": r.code, "kind": r.kind, "enabled": bool(r.enabled),
             "threshold": r.threshold, "weight": r.weight}
            for r in sorted(get_rules(db).values(), key=lambda r: int(r.code[1:]))]


def rules_from_snapshot(snapshot: list) -> dict:
    from types import SimpleNamespace
    return {r["code"]: SimpleNamespace(**r) for r in snapshot}


@dataclass
class RuleEvaluation:
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    penalty: float = 0.0
    requires_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    applied_rule_ids: List[str] = field(default_factory=list)
    policy_explanation: List[str] = field(default_factory=list)


def evaluate_candidate(
    db: Session,
    *,
    skill_match: float,          # 0-100
    on_leave_or_absent: bool,
    load_ratio: float,           # e.g. 0.0-2.0
    recovery_risk: float,        # 0-1
    high_complexity_recent: int,
    security_critical_task: bool,
    has_prior_ownership: bool,
    fairness_deviation: float,   # candidate's deviation from team median load, as a ratio
    emergency_mode: bool = False,
) -> RuleEvaluation:
    return evaluate_with_rules(
        get_rules(db), skill_match=skill_match, on_leave_or_absent=on_leave_or_absent,
        load_ratio=load_ratio, recovery_risk=recovery_risk, high_complexity_recent=high_complexity_recent,
        security_critical_task=security_critical_task, has_prior_ownership=has_prior_ownership,
        fairness_deviation=fairness_deviation, emergency_mode=emergency_mode,
    )


def evaluate_with_rules(
    rules: dict, *, skill_match: float, on_leave_or_absent: bool, load_ratio: float,
    recovery_risk: float, high_complexity_recent: float, security_critical_task: bool,
    has_prior_ownership: bool, fairness_deviation: float, emergency_mode: bool = False,
) -> RuleEvaluation:
    """Pure rule evaluation against a given rule set (DB rows or a receipt snapshot)."""
    ev = RuleEvaluation()

    def rule(code):
        r = rules.get(code)
        return r if r and r.enabled else None

    r1 = rule("R1")
    if r1 and skill_match < (r1.threshold or 70.0):
        ev.excluded = True
        ev.exclusion_reason = f"R1: skill confidence {skill_match:.0f}% is below the required {r1.threshold:.0f}%"
        ev.applied_rule_ids.append("R1")
        return ev
    elif r1:
        ev.policy_explanation.append(f"R1 passed: skill confidence {skill_match:.0f}% meets the {r1.threshold:.0f}% threshold")
        ev.applied_rule_ids.append("R1")

    r2 = rule("R2")
    if r2 and on_leave_or_absent:
        ev.excluded = True
        ev.exclusion_reason = "R2: candidate is on approved leave or already marked absent"
        ev.applied_rule_ids.append("R2")
        return ev
    elif r2:
        ev.policy_explanation.append("R2 passed: no leave/absence overlap")
        ev.applied_rule_ids.append("R2")

    r3 = rule("R3")
    if r3 and load_ratio > (r3.threshold or 1.0) and not emergency_mode:
        ev.excluded = True
        ev.exclusion_reason = f"R3: workload ratio {load_ratio:.2f} exceeds the {r3.threshold:.2f} ceiling"
        ev.applied_rule_ids.append("R3")
        return ev
    elif r3:
        ev.policy_explanation.append(f"R3 passed: workload ratio {load_ratio:.2f} is within the {r3.threshold:.2f} ceiling")
        ev.applied_rule_ids.append("R3")

    r4 = rule("R4")
    if r4 and recovery_risk > (r4.threshold or 0.75):
        # severe penalty rather than hard exclusion, per spec ("severe penalty or exclusion")
        ev.penalty += (r4.weight or 30.0)
        ev.applied_rule_ids.append("R4")
        ev.policy_explanation.append(
            f"R4 triggered: recovery risk {recovery_risk:.2f} exceeds {r4.threshold:.2f} — severe score penalty applied"
        )
    elif r4:
        ev.policy_explanation.append(f"R4 passed: recovery risk {recovery_risk:.2f} is within {r4.threshold:.2f}")
        ev.applied_rule_ids.append("R4")

    r5 = rule("R5")
    if r5 and security_critical_task and not has_prior_ownership:
        ev.requires_review = True
        ev.review_reasons.append("R5: security-critical task assigned to someone without prior ownership")
        ev.applied_rule_ids.append("R5")

    r6 = rule("R6")
    if r6 and high_complexity_recent >= (r6.threshold or 2.0):
        ev.penalty += (r6.weight or 10.0)
        ev.applied_rule_ids.append("R6")
        ev.policy_explanation.append(
            f"R6 triggered: {high_complexity_recent} high-complexity tasks in the last 48h — workload penalty applied"
        )

    r9 = rule("R9")
    if r9 and fairness_deviation > (r9.threshold or 1.5):
        ev.penalty += (r9.weight or 15.0)
        ev.applied_rule_ids.append("R9")
        ev.policy_explanation.append(
            f"R9 triggered: allocation would push workload {fairness_deviation:.2f}x above team median — fairness penalty applied"
        )

    return ev


def confidence_label(confidence_0_100: float, db: Session) -> Optional[str]:
    return confidence_label_with_rules(confidence_0_100, get_rules(db))


def confidence_label_with_rules(confidence_0_100: float, rules: dict) -> Optional[str]:
    r10 = rules.get("R10")
    threshold = (r10.threshold if r10 and r10.enabled else 0.60) * 100
    if confidence_0_100 < threshold:
        return "Insufficient Confidence"
    return None
