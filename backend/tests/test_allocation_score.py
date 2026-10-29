"""AllocationScore v2 (paper Section IV-G): reproduces Tables 8-10 from the shipped code."""
import itertools

import numpy as np
import pytest

from app.ai import assignment_engine as ae


def _args(sk, hs, cap, L, RR, kappa=0.0, fa=1.0, ur=0.5, penalty=0):
    return dict(skill_match=sk, historical_success=hs, availability=cap,
                completion_conf=ae.completion_confidence(sk * 100, hs * 100, cap * 100),
                fairness=fa, urgency_fit=ur, load_ratio=L, recovery_risk=RR,
                rule_penalty_points=penalty, transfer_cost=kappa)


ARCHETYPES = {"strong": (.9, .9, .8), "average": (.7, .7, .5), "weak": (.5, .5, .3)}
LOADS = [(.3, .2), (.6, .4), (.8, .5), (1.0, .7)]
TABLE8 = {  # (v1 row, v2 row) as printed in the paper
    "strong": ([60.2, 34.7, 19.2, 0.0], [78.1, 70.4, 65.8, 59.8]),
    "average": ([42.1, 16.6, 1.1, 0.0], [60.0, 52.3, 47.7, 41.7]),
    "weak": ([26.0, 0.5, 0.0, 0.0], [43.8, 36.2, 31.5, 25.5]),
}


@pytest.mark.parametrize("name", list(ARCHETYPES))
def test_table8_reproduced(name):
    sk, hs, cap = ARCHETYPES[name]
    v1 = [round(ae.allocation_score_v1(**_args(sk, hs, cap, L, R)), 1) for L, R in LOADS]
    v2 = [round(ae.allocation_score_v2(**_args(sk, hs, cap, L, R))["display"], 1) for L, R in LOADS]
    assert v1 == TABLE8[name][0]
    assert v2 == TABLE8[name][1]


def _grid():
    for sk, hs, cap, L, R in itertools.product([.5, .7, .9], [.5, .7, .9], [.2, .5, .8],
                                                [.2, .5, .8, 1.0], [.1, .3, .5, .7]):
        yield _args(sk, hs, cap, L, R, kappa=0.2)


def test_reference_grid_and_table9():
    grid = list(_grid())
    assert len(grid) == 432
    v1 = np.array([ae.allocation_score_v1(**a) for a in grid])
    v2 = np.array([ae.allocation_score_v2(**a)["display"] for a in grid])
    assert round(100 * (v1 == 0).mean(), 1) == 30.8 and round(v1.mean(), 1) == 16.0
    assert (v2 == 0).sum() == 0 and round(v2.mean(), 1) == 48.9
    assert (round(v2.min(), 1), round(v2.max(), 1)) == (20.6, 78.1)
    table9 = {35: (16.0, 90.0), 45: (6.9, 64.4), 55: (1.9, 28.5)}
    for cut, (p1, p2) in table9.items():
        assert round(100 * (v1 >= cut).mean(), 1) == p1
        assert round(100 * (v2 >= cut).mean(), 1) == p2


def test_table10_fairness_counted_once():
    for r, v2_pts, v1_total in [(0, 0.0, 0.0), (1, 0.8, 8.8), (2, 1.6, 17.6), (3, 2.4, 26.4), (4, 3.2, 28.2)]:
        fa_pts = round(10 * (1 - ae.fairness_term(1.0, r)), 1)
        assert fa_pts == v2_pts
        assert round(fa_pts + min(25, 8 * r), 1) == v1_total


def test_workload_penalty_bounded_by_30_points():
    worst = ae.allocation_score_v2(**_args(1, 1, 1, L=5.0, RR=1.0))
    assert worst["breakdown"]["workload_penalty"] == -30.0


def test_penalty_balance_principle():
    largest_positive = max(ae.POSITIVE_WEIGHTS.values())
    assert ae.WORKLOAD_LAMBDA <= largest_positive and ae.TRANSFER_WEIGHT <= largest_positive
    assert abs(sum(ae.POSITIVE_WEIGHTS.values()) - 1.0) < 1e-12


def test_breakdown_sums_to_score_when_unclipped():
    s = ae.allocation_score_v2(**_args(.8, .7, .6, .5, .4, kappa=0.3, penalty=10))
    assert not s["clipped"]
    assert abs(sum(s["breakdown"].values()) - s["display"]) < 0.05


def test_rule_penalties_can_still_override():
    s = ae.allocation_score_v2(**_args(.9, .9, .8, .3, .2, penalty=55))
    assert s["display"] < ae.MIN_VIABLE_SCORE
