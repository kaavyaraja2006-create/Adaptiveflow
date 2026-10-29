"""Rule-engine tests (paper Table 5): 16 tests, boundary values explicit."""
from app.ai import rules_engine


def ev(db, **kw):
    base = dict(skill_match=85, on_leave_or_absent=False, load_ratio=0.5, recovery_risk=0.2,
                high_complexity_recent=0, security_critical_task=False, has_prior_ownership=True,
                fairness_deviation=1.0, emergency_mode=False)
    base.update(kw)
    return rules_engine.evaluate_candidate(db, **base)


# R1 minimum skill confidence (strict <)
def test_r1_excludes_below_70(db):
    r = ev(db, skill_match=69.9)
    assert r.excluded and r.exclusion_reason.startswith("R1")


def test_r1_boundary_70_passes(db):
    assert not ev(db, skill_match=70).excluded


# R2 leave / absent
def test_r2_excludes_absent(db):
    r = ev(db, on_leave_or_absent=True)
    assert r.excluded and r.exclusion_reason.startswith("R2")


# R3 workload ceiling (strict >), emergency bypass
def test_r3_excludes_above_1(db):
    r = ev(db, load_ratio=1.01)
    assert r.excluded and r.exclusion_reason.startswith("R3")


def test_r3_boundary_1_passes(db):
    assert not ev(db, load_ratio=1.0).excluded


def test_r3_emergency_bypass(db):
    assert not ev(db, load_ratio=1.5, emergency_mode=True).excluded


# R4 severe recovery risk (strict >)
def test_r4_penalty_above_075(db):
    r = ev(db, recovery_risk=0.76)
    assert r.penalty == 30 and "R4" in r.applied_rule_ids and not r.excluded


def test_r4_boundary_075_no_penalty(db):
    assert ev(db, recovery_risk=0.75).penalty == 0


# R5 security-critical ownership -> review, no penalty
def test_r5_review_flag(db):
    r = ev(db, security_critical_task=True, has_prior_ownership=False)
    assert r.requires_review and r.penalty == 0


# R6 recent high-complexity load (>= 2)
def test_r6_penalty_at_2(db):
    assert ev(db, high_complexity_recent=2).penalty == 10


def test_r6_no_penalty_at_1(db):
    assert ev(db, high_complexity_recent=1).penalty == 0


# R7 defined but not evaluated by any code path (documented limitation)
def test_r7_defined_not_evaluated(db):
    assert "R7" in rules_engine.get_rules(db)
    r = ev(db, skill_match=70)  # weakest passing match
    assert "R7" not in r.applied_rule_ids


# R9 fairness ceiling (strict >)
def test_r9_penalty_above_15(db):
    assert ev(db, fairness_deviation=1.51).penalty == 15


def test_r9_boundary_15_no_penalty(db):
    assert ev(db, fairness_deviation=1.5).penalty == 0


# Stacking of R4 + R6 + R9 = 55 points
def test_penalty_stacking_55(db):
    r = ev(db, recovery_risk=0.9, high_complexity_recent=3, fairness_deviation=2.0)
    assert r.penalty == 55


# R10 insufficient confidence (< 60)
def test_r10_label_boundary(db):
    assert rules_engine.confidence_label(59, db) == "Insufficient Confidence"
    assert rules_engine.confidence_label(60, db) is None
