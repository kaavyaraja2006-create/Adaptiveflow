"""
Human-Aware Allocation Engine (upgraded).

Given a piece of work and a pool of candidates, this module:

1. Runs symbolic rules (R1-R10, rules_engine.py) FIRST — hard exclusions
   always sit above scoring. A high-scoring candidate excluded by a rule
   stays excluded.
2. Computes Sustainable Workload / Recovery Risk for each surviving
   candidate (workload_engine.py) — never a medical/emotional score, and
   employees who've opted out are skipped entirely.
3. Computes a knowledge-transfer cost penalty for candidates without
   strong domain familiarity/ownership of the relevant work.
4. Combines everything into a single hybrid AllocationScore (v2, paper
   Section IV-G, Equations 2-5), on a 0-1 scale and displayed x100:

   S   = clip01( Pos - lambda_W * W - P/100 - 0.15 * kappa )
   Pos = 0.30*SkillMatch + 0.20*HistoricalSuccess + 0.15*Availability
         + 0.15*CompletionConfidence + 0.10*Fairness + 0.10*UrgencyFit
   W   = 0.55*min(1, LoadRatio) + 0.45*RecoveryRisk,   lambda_W = 0.30
   P   = total rule penalty in points (R4 30, R6 10, R9 15; at most 55)

   Changes from v1: (a) the workload term is scaled by lambda_W = 0.30, so it
   can remove at most 30 points instead of 100 (penalty-balance principle:
   no smooth penalty outweighs the largest positive weight, 0.30); (b) the
   second-stage deduction min(25, 8*r) for recent reassignments is gone, so
   fairness enters only once, through the Fairness term. v1 is kept in
   `allocation_score_v1` for the equation audit only; it is never served.

5. Returns the top candidates with both a model explanation (which
   quantitative factors helped/hurt) and a policy explanation (which rules
   fired and why) — never a single opaque number.

Backward compatible: every field the original engine returned
(skill, success_rate, capacity, completion_probability, fairness_penalty,
overall, history_count) is still present, so the existing frontend keeps
working unmodified. New fields are additive.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from .. import models
from . import capability_engine, workload_engine, rules_engine, data_quality

SPRINT_CAPACITY_POINTS = 40.0
MIN_VIABLE_SCORE = 45   # a-priori policy constant, not fitted to outcomes (paper Table 9)
MIN_CAPACITY = 15

SCORE_VERSION = "allocation-score-v2"
POSITIVE_WEIGHTS = {"skill_match": 0.30, "historical_success": 0.20, "availability": 0.15,
                    "completion_confidence": 0.15, "fairness": 0.10, "urgency_fit": 0.10}
WORKLOAD_LAMBDA = 0.30          # v1 was effectively 1.0
WORKLOAD_LOAD_WEIGHT = 0.55
WORKLOAD_RISK_WEIGHT = 0.45
TRANSFER_WEIGHT = 0.15
REASSIGNMENT_FAIRNESS_STEP = 0.08
URGENCY_MAP = {"Critical": 1.0, "High": 0.75, "Medium": 0.5, "Low": 0.25}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def completion_confidence(skill: float, success_rate: float, capacity: float) -> float:
    """Cc on a 0-1 scale (a weighted mix, not an independent forecast)."""
    return (0.35 * skill + 0.40 * success_rate + 0.25 * min(100.0, 1.2 * capacity)) / 100.0


def fairness_term(fairness_deviation: float, recent_reassignments: int) -> float:
    """Fa = clip01(1 - min(1, delta - 1) - 0.08 r). For delta <= 1 the middle term is <= 0
    and is floored by the outer clip, exactly as in the v1 code."""
    return _clip01(1 - min(1.0, fairness_deviation - 1) - REASSIGNMENT_FAIRNESS_STEP * recent_reassignments)


def workload_term(load_ratio: float, recovery_risk: float) -> float:
    return WORKLOAD_LOAD_WEIGHT * min(1.0, load_ratio) + WORKLOAD_RISK_WEIGHT * recovery_risk


def allocation_score_v2(*, skill_match: float, historical_success: float, availability: float,
                        completion_conf: float, fairness: float, urgency_fit: float,
                        load_ratio: float, recovery_risk: float, rule_penalty_points: float,
                        transfer_cost: float) -> dict:
    """Pure function for Equations (2)-(4). All signal inputs on 0-1; returns score and breakdown in points."""
    signals = {"skill_match": skill_match, "historical_success": historical_success,
               "availability": availability, "completion_confidence": completion_conf,
               "fairness": fairness, "urgency_fit": urgency_fit}
    positive = {k: POSITIVE_WEIGHTS[k] * v for k, v in signals.items()}
    pos = sum(positive.values())
    w_pen = WORKLOAD_LAMBDA * workload_term(load_ratio, recovery_risk)
    r_pen = rule_penalty_points / 100.0
    t_pen = TRANSFER_WEIGHT * transfer_cost
    raw = pos - w_pen - r_pen - t_pen
    score = _clip01(raw)
    breakdown = {k: round(100 * v, 2) for k, v in positive.items()}
    breakdown.update({"workload_penalty": round(-100 * w_pen, 2), "rule_penalty": round(-100 * r_pen, 2),
                      "transfer_cost": round(-100 * t_pen, 2)})
    return {"score": score, "raw": raw, "display": 100 * score, "breakdown": breakdown,
            "clipped": raw < 0 or raw > 1, "version": SCORE_VERSION}


def allocation_score_v1(*, skill_match, historical_success, availability, completion_conf, fairness,
                        urgency_fit, load_ratio, recovery_risk, rule_penalty_points, transfer_cost,
                        recent_reassignments: int = 0) -> float:
    """LEGACY v1 score (0-100), kept only so the equation audit can reproduce Tables 8-10.
    Workload coefficient 1.0 and a second-stage min(25, 8r) deduction. Never served."""
    pos = (0.30 * skill_match + 0.20 * historical_success + 0.15 * availability
           + 0.15 * completion_conf + 0.10 * fairness + 0.10 * urgency_fit)
    s = _clip01(pos - workload_term(load_ratio, recovery_risk) - rule_penalty_points / 100.0
                - TRANSFER_WEIGHT * transfer_cost) * 100
    if recent_reassignments:
        s = max(0.0, min(100.0, s - min(25, 8 * recent_reassignments)))
    return s


def _recent_reassignment_count(db: Session, employee_id: int, days: int = 14) -> int:
    cutoff = datetime.utcnow() - timedelta(days=days)
    return (
        db.query(models.DecisionHistoryEntry)
        .filter(models.DecisionHistoryEntry.recommendation.like(f"%employee_id={employee_id}%"))
        .filter(models.DecisionHistoryEntry.created_at >= cutoff)
        .count()
    )


def _active_points_for(db: Session, employee_id: int) -> float:
    issues = (
        db.query(models.Issue)
        .filter(models.Issue.assignee_id == employee_id, models.Issue.status != "Done")
        .all()
    )
    return sum(i.story_points or 0 for i in issues)


def _team_median_active_points(db: Session, employee_ids: List[int]) -> float:
    if not employee_ids:
        return 0.0
    values = sorted(_active_points_for(db, eid) for eid in employee_ids)
    n = len(values)
    mid = n // 2
    if n % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2 if n else 0.0
    return values[mid]


def _domain_familiarity(candidate: models.Employee, issue: models.Issue) -> float:
    """0-1 heuristic: has this person worked on this project/task type before?
    Approximated from real history rather than fabricated — but explicitly
    labeled as a heuristic in the explanation, since we don't track true
    code ownership (no Git integration in this build)."""
    same_type = [h for h in candidate.history if h.task_type == issue.issue_type]
    if not same_type:
        return 0.15
    return min(1.0, 0.3 + len(same_type) * 0.08)


def _has_prior_ownership(candidate: models.Employee, issue: models.Issue) -> bool:
    return any(h.task_type == issue.issue_type for h in candidate.history)


def transfer_cost_from(n_same_type: int, story_points: Optional[int]) -> float:
    """Equation (5), from the two quantities it actually depends on (pure, replayable)."""
    familiarity = 0.15 if n_same_type == 0 else min(1.0, 0.3 + n_same_type * 0.08)
    ownership = 1.0 if n_same_type > 0 else 0.2
    complexity = min(1.0, (story_points or 3) / 13.0)
    penalty = 0.4 * (1 - familiarity) + 0.35 * (1 - ownership) + 0.25 * complexity
    return round(max(0.0, min(1.0, penalty)), 3)


def _transfer_penalty(candidate: models.Employee, issue: models.Issue) -> float:
    n = sum(1 for h in candidate.history if h.task_type == issue.issue_type)
    return transfer_cost_from(n, issue.story_points)


def _f(x):
    """Plain float (or None) so receipts round-trip through canonical JSON exactly."""
    return None if x is None else float(x)


def task_inputs(issue: models.Issue) -> dict:
    security_critical = issue.issue_type == "Security" or any(
        (lbl.name if hasattr(lbl, "name") else lbl) == "security" for lbl in (issue.labels or [])
    )
    return {"task_id": issue.id, "issue_type": issue.issue_type, "priority": issue.priority,
            "story_points": issue.story_points, "security_critical": bool(security_critical)}


def gather_candidate_inputs(db: Session, candidate: models.Employee, issue: models.Issue,
                            team_median_points: float) -> Optional[dict]:
    """Everything the score depends on, read from the database. This is what a decision
    receipt stores, so the decision can later be recomputed without the database."""
    got = _gather(db, candidate, issue, team_median_points)
    return None if got is None else got[0]


def _gather(db: Session, candidate: models.Employee, issue: models.Issue, team_median_points: float):
    """(inputs, gate summary) — one workload computation per candidate."""
    if candidate.workload_opt_out:
        return None  # opted-out employees are never scored/considered
    profile = capability_engine.capability_profile(candidate)
    breakdown = profile["breakdown"].get(issue.issue_type, {"skill": 45, "success_rate": 50, "completed_tasks": 0})
    wprofile = workload_engine.workload_profile(db, candidate)
    active_points = _active_points_for(db, candidate.id)
    return {
        "employee_id": candidate.id,
        "skill": breakdown["skill"], "success_rate": breakdown["success_rate"],
        "capacity": profile["available_capacity"], "history_count": breakdown["completed_tasks"],
        "n_same_type_history": sum(1 for h in candidate.history if h.task_type == issue.issue_type),
        "on_leave_or_absent": candidate.status == "absent",
        "active_points": _f(active_points), "team_median_points": _f(team_median_points),
        "recent_reassignments": _recent_reassignment_count(db, candidate.id),
        "load_ratio": _f(wprofile.load_ratio),
        "high_complexity_density": _f(wprofile.high_complexity_density),
        "context_switching": _f(wprofile.context_switching),
        # recovery risk: the value served downstream, plus what the gate chose between
        "recovery_risk_served": _f(wprofile.recovery_risk),
        "recovery_risk_baseline": _f(wprofile.baseline_recovery_risk),
        "recovery_risk_model": _f(wprofile.model_recovery_risk),
        "recovery_risk_band": wprofile.risk_band,
        "recovery_risk_today": _f(wprofile.current_day_risk),
        "recovery_risk_gate_status": (wprofile.gate or {}).get("applied_status"),
        "workload_model_status": wprofile.model_status,
    }, wprofile.gate


def score_from_inputs(inp: dict, task: dict, rules: dict, emergency_mode: bool = False,
                      name: Optional[str] = None, wextra: Optional[dict] = None) -> dict:
    """Pure scoring of one candidate: rules (from a rule set) then AllocationScore v2.
    Used both live and by the receipt replay, so the two can never diverge."""
    skill, success_rate, capacity = inp["skill"], inp["success_rate"], inp["capacity"]
    load_ratio, recovery_risk = inp["load_ratio"], inp["recovery_risk_served"]
    median = inp["team_median_points"]
    fairness_deviation = (inp["active_points"] / median) if median and median > 0 else 1.0
    has_ownership = inp["n_same_type_history"] > 0

    rule_eval = rules_engine.evaluate_with_rules(
        rules,
        skill_match=skill, on_leave_or_absent=inp["on_leave_or_absent"], load_ratio=load_ratio,
        recovery_risk=recovery_risk,
        high_complexity_recent=inp["high_complexity_density"] * 3,  # back out an approx recent count
        security_critical_task=task["security_critical"], has_prior_ownership=has_ownership,
        fairness_deviation=fairness_deviation, emergency_mode=emergency_mode,
    )
    fairness_hits = inp["recent_reassignments"]
    completion_confidence_n = completion_confidence(skill, success_rate, capacity)
    completion_probability = round(100 * completion_confidence_n)   # display value only
    transfer_cost = transfer_cost_from(inp["n_same_type_history"], task["story_points"])

    skill_match_n = skill / 100
    historical_success_n = success_rate / 100
    availability_n = min(1.0, capacity / 100)
    fairness_n = fairness_term(fairness_deviation, fairness_hits)
    urgency_fit_n = URGENCY_MAP.get(task["priority"], 0.5)

    scored = allocation_score_v2(
        skill_match=skill_match_n, historical_success=historical_success_n, availability=availability_n,
        completion_conf=completion_confidence_n, fairness=fairness_n, urgency_fit=urgency_fit_n,
        load_ratio=load_ratio, recovery_risk=recovery_risk, rule_penalty_points=rule_eval.penalty,
        transfer_cost=transfer_cost,
    )
    overall = round(scored["display"])
    # Points the Fairness term gives up relative to a perfectly balanced candidate. In v2 this is the
    # ONLY place fairness affects the score (v1 also subtracted min(25, 8r) a second time).
    fairness_penalty_pts = round(100 * POSITIVE_WEIGHTS["fairness"] * (1 - fairness_n))

    top_positive, top_negative = [], []
    factors = [
        ("skill match", skill_match_n, f"{skill}% skill match"),
        ("historical success", historical_success_n, f"{success_rate}% historical success rate"),
        ("availability", availability_n, f"{capacity}% free capacity"),
        ("completion confidence", completion_confidence_n, f"{completion_probability}% predicted completion confidence"),
        ("fairness", fairness_n, "balanced recent workload" if fairness_n > 0.6 else "above-median recent workload"),
        ("urgency fit", urgency_fit_n, f"{task['priority']} priority task"),
    ]
    for _, val, label in sorted(factors, key=lambda f: -f[1])[:3]:
        if val >= 0.5:
            top_positive.append(label)
    ctx = inp["context_switching"]
    negatives = [
        ("existing workload", load_ratio, f"load ratio {load_ratio:.2f}"),
        ("recovery risk", recovery_risk, f"recovery risk {recovery_risk:.2f} ({inp['recovery_risk_band']})"),
        ("context switching", ctx, f"{ctx:.1f} active tasks/day recently"),
        ("transfer cost", transfer_cost, f"transfer cost {transfer_cost:.2f} (limited prior exposure)" if transfer_cost > 0.4 else None),
    ]
    for label, val, text in negatives:
        if text and (val >= 0.5 if label != "context switching" else val >= 2):
            top_negative.append(text)

    wextra = wextra or {}
    return {
        "employee_id": inp["employee_id"],
        "name": name,
        "skill": skill,
        "success_rate": success_rate,
        "capacity": capacity,
        "completion_probability": min(99, max(5, completion_probability)),
        "fairness_penalty": fairness_penalty_pts,
        "overall": overall,                    # v2: no second-stage deduction
        "history_count": inp["history_count"],
        # ---- extended, additive fields ----
        "workload_ratio": round(load_ratio, 2),
        "recovery_risk": recovery_risk,
        "baseline_recovery_risk": inp["recovery_risk_baseline"],
        "model_recovery_risk": inp["recovery_risk_model"],
        "recovery_risk_gate": wextra.get("gate"),
        "recovery_risk_band": inp["recovery_risk_band"],
        "workload_model_status": inp["workload_model_status"],
        "allocation_score": overall,
        "score_version": scored["version"],
        "score_breakdown": scored["breakdown"],
        "recent_reassignments": fairness_hits,
        "transfer_cost": transfer_cost,
        "excluded_by_rule": rule_eval.excluded,
        "exclusion_reason": rule_eval.exclusion_reason,
        "requires_manager_review": rule_eval.requires_review,
        "review_reasons": rule_eval.review_reasons,
        "applied_rule_ids": rule_eval.applied_rule_ids,
        "policy_explanation": rule_eval.policy_explanation,
        "model_explanation": {"top_positive_factors": top_positive, "top_negative_factors": top_negative},
    }


def score_candidate(db: Session, candidate: models.Employee, issue: models.Issue,
                     team_median_points: float = 0.0, emergency_mode: bool = False) -> Optional[dict]:
    got = _gather(db, candidate, issue, team_median_points)
    if got is None:
        return None
    inp, gate = got
    return score_from_inputs(inp, task_inputs(issue), rules_engine.get_rules(db), emergency_mode,
                             name=candidate.name, wextra={"gate": gate})


def rank_candidates(scored: List[dict]) -> dict:
    """Pure selection step: rule exclusions, sort by score (stable, so ties keep input order),
    viability cutoff. Returns the ordered lists and the top viable candidate (or None)."""
    rule_excluded = [s for s in scored if s["excluded_by_rule"]]
    eligible = sorted((s for s in scored if not s["excluded_by_rule"]), key=lambda s: -s["overall"])
    viable = [s for s in eligible if s["overall"] >= MIN_VIABLE_SCORE and s["capacity"] >= MIN_CAPACITY]
    return {"eligible": eligible, "rule_excluded": rule_excluded, "viable": viable,
            "best": viable[0] if viable else None}


def concurrent_absence_adjustment(candidates_top5: List[dict], recommended: dict, prior_picks: int) -> Optional[dict]:
    """Pure version of the spread-across-concurrent-absences rule: if the top pick already received
    a task in this batch, switch to the runner-up when it is eligible, viable and within 10 points."""
    if prior_picks <= 0 or len(candidates_top5) <= 1:
        return None
    alt = candidates_top5[1]
    if alt and not alt.get("excluded_by_rule") and alt["overall"] >= MIN_VIABLE_SCORE \
            and (recommended["overall"] - alt["overall"]) <= 10:
        return alt
    return None


def recommend_for_issue(db: Session, issue: models.Issue, exclude_employee_id: Optional[int] = None,
                         emergency_mode: bool = False) -> dict:
    from . import decision_receipt
    dq_report = data_quality.check_issue(db, issue)

    candidates_q = db.query(models.Employee).filter(models.Employee.status == "available")
    if exclude_employee_id:
        candidates_q = candidates_q.filter(models.Employee.id != exclude_employee_id)
    candidates = candidates_q.order_by(models.Employee.id.asc()).all()
    candidate_ids = [c.id for c in candidates]
    team_median_points = _team_median_active_points(db, candidate_ids)

    rules = rules_engine.get_rules(db)
    task = task_inputs(issue)
    inputs, scored, opted_out = [], [], []
    for c in candidates:
        got = _gather(db, c, issue, team_median_points)
        if got is None:
            opted_out.append(c.id)
            continue
        inp, gate = got
        inputs.append(inp)
        scored.append(score_from_inputs(inp, task, rules, emergency_mode, name=c.name, wextra={"gate": gate}))

    ranked = rank_candidates(scored)
    scored_eligible, rule_excluded, best = ranked["eligible"], ranked["rule_excluded"], ranked["best"]

    base_result = {
        "task_id": issue.id,
        "task_title": issue.title,
        "data_quality_flags": dq_report.flags,
        "rule_excluded_count": len(rule_excluded),
    }

    if best is None:
        result = {
            **base_result,
            "no_suitable_candidate": True,
            "recommended_employee_id": None,
            "recommended_employee_name": None,
            "confidence": None,
            "confidence_label": "Insufficient Confidence",
            "reason": (
                "No available team member currently has both sufficient skill match and free capacity "
                "for this task, once policy rules and workload/recovery-risk limits are applied (R8: "
                "no eligible candidate)."
            ),
            "alternatives": [
                "Split the task into smaller pieces across two people",
                "Move the deadline out",
                "Reduce scope of the task",
                "Request an additional resource for this team",
            ],
            "candidates": (scored_eligible + rule_excluded)[:5],
            "model_explanation": None,
            "policy_explanation": [f"{s['name']}: {s['exclusion_reason']}" for s in rule_excluded[:5]],
        }
    else:
        reason = (
            f"{best['name']} has a {best['success_rate']}% historical success rate on "
            f"{issue.issue_type} work across {best['history_count']} comparable tasks, "
            f"a {best['skill']}% skill match, and {best['capacity']}% free capacity."
        )
        if best["fairness_penalty"] > 0:
            reason += (f" The fairness term gave up {best['fairness_penalty']} of its 10 points "
                       "(above-median recent load or recent reassignments).")
        if best["recovery_risk_band"] in ("elevated", "severe"):
            reason += f" Note: {best['name']}'s recovery risk is currently {best['recovery_risk_band']}."

        result = {
            **base_result,
            "no_suitable_candidate": False,
            "recommended_employee_id": best["employee_id"],
            "recommended_employee_name": best["name"],
            "confidence": best["overall"],
            "confidence_label": rules_engine.confidence_label_with_rules(best["overall"], rules),
            "reason": reason,
            "alternatives": [],
            "candidates": (scored_eligible + rule_excluded)[:5],
            "model_explanation": best["model_explanation"],
            "policy_explanation": best["policy_explanation"],
            "requires_manager_review": best["requires_manager_review"],
            "review_reasons": best["review_reasons"],
            "transfer_cost": best["transfer_cost"],
        }

    result["receipt"] = decision_receipt.build_receipt(
        db, task=task, inputs=inputs, scored=scored, result=result, opted_out_ids=opted_out,
        context={"emergency_mode": bool(emergency_mode), "exclude_employee_id": exclude_employee_id,
                 "trigger": "recommend_for_issue", "prior_picks_of_top": 0},
    )
    return result


def handle_absence(db: Session, employee_id: int, tentative_load: dict = None) -> List[dict]:
    """Core signature flow: mark someone absent, find every task assigned to
    them, and produce a ranked, explainable reassignment recommendation for
    each one — optimizing across the whole affected set rather than greedily
    giving everything to whoever looks best in isolation.

    `tentative_load` can be passed in and shared across multiple calls (one
    per absent employee) so that when several people are absent at once,
    fairness is balanced across the WHOLE batch, not just within each
    individual's own tasks — otherwise every absent person's tasks would
    independently pick the same single best-available person."""
    from . import decision_receipt
    affected_issues = (
        db.query(models.Issue)
        .filter(models.Issue.assignee_id == employee_id)
        .filter(models.Issue.status != "Done")
        .order_by(models.Issue.id.asc())
        .all()
    )

    results = []
    if tentative_load is None:
        tentative_load = {}

    for issue in affected_issues:
        rec = recommend_for_issue(db, issue, exclude_employee_id=employee_id)
        rec["receipt"]["context"]["trigger"] = "employee_absence"
        rec["receipt"]["context"]["absent_employee_id"] = employee_id
        if not rec["no_suitable_candidate"]:
            top = next(c for c in rec["candidates"] if c["employee_id"] == rec["recommended_employee_id"])
            picks = tentative_load.get(rec["recommended_employee_id"], 0)
            rec["receipt"]["context"]["prior_picks_of_top"] = picks
            alt = concurrent_absence_adjustment(rec["candidates"], top, picks)
            if alt is not None:
                alt_reason = (
                    f"{alt['name']} has a {alt['success_rate']}% historical success rate on "
                    f"{issue.issue_type} work across {alt['history_count']} comparable tasks, "
                    f"a {alt['skill']}% skill match, and {alt['capacity']}% free capacity."
                )
                rec["recommended_employee_id"] = alt["employee_id"]
                rec["recommended_employee_name"] = alt["name"]
                rec["confidence"] = alt["overall"]
                # label must describe the candidate actually recommended (the old code kept the top pick's label)
                rec["confidence_label"] = rules_engine.confidence_label(alt["overall"], db)
                rec["reason"] = (
                    f"Assigned to {alt['name']} instead of the top-ranked candidate to spread workload "
                    f"across concurrent absences. {alt_reason}"
                )
                rec["model_explanation"] = alt["model_explanation"]
                rec["policy_explanation"] = alt["policy_explanation"]
            decision_receipt.update_decision(rec["receipt"], rec)
            tentative_load[rec["recommended_employee_id"]] = tentative_load.get(rec["recommended_employee_id"], 0) + 1
        results.append(rec)
    return results
