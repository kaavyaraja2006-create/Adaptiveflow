"""Segment-wise baseline gate (novelty #1)."""
import json
import os

import numpy as np
import pytest

from app.ai import baseline_gate as bg


def test_holm_step_down():
    p = {"a": 0.001, "b": 0.010, "c": 0.013, "d": 0.5}
    # thresholds at alpha=0.025: a<=0.00625 yes, b<=0.00833 no -> stop
    assert bg.holm(p) == {"a": True, "b": False, "c": False, "d": False}
    assert bg.holm({"a": 0.001, "b": 0.006}) == {"a": True, "b": True}


def test_segment_functions_use_prediction_time_information():
    assert [bg.duration_segment(x) for x in (1, 2, 3, 5, 8, 13, 20)] == \
        ["sp_1_2", "sp_1_2", "sp_3_5", "sp_3_5", "sp_6_13", "sp_6_13", "sp_14_plus"]
    assert [bg.risk_segment(x) for x in (0.1, 0.3, 0.55, 0.75, 0.9)] == \
        ["low", "moderate", "elevated", "severe", "severe"]


def _data(seed=0, n_clusters=60, per=40, win_segment="sp_14_plus", effect=3.0):
    """Model equals baseline everywhere except one segment, where it is much better."""
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_clusters), per)
    sp = rng.choice([1, 3, 8, 20], len(clusters))
    y = 10 + rng.normal(0, 3, n_clusters)[clusters] + rng.normal(0, 2, len(clusters))
    base = y + rng.normal(0, effect, len(y))
    model = y + rng.normal(0, effect, len(y))
    seg = np.array([bg.duration_segment(s) for s in sp])
    win = seg == win_segment
    model[win] = y[win] + rng.normal(0, 0.3, win.sum())
    return y, model, base, clusters, seg


def test_detects_the_one_segment_where_the_model_wins():
    y, m, b, c, seg = _data()
    rec = bg.evaluate_gate_by_segment("duration", "m", "b", y, m, b, c, seg, "synthetic")
    st = {k: v["status"] for k, v in rec["segments"].items()}
    assert st["sp_14_plus"] == bg.PROMOTED
    assert all(v == bg.DEMOTED for k, v in st.items() if k != "sp_14_plus")
    # the policy beats always-baseline and always-model on this data
    pol = bg.segment_policy_mae(y, m, b, seg, rec)
    assert pol < np.mean(np.abs(y - b)) and pol < np.mean(np.abs(y - m))


def test_holm_controls_false_promotions_under_the_null():
    """Model and baseline identical in distribution in every segment: promotions should be rare."""
    false = 0
    for s in range(20):
        y, m, b, c, seg = _data(seed=100 + s, win_segment="none")
        rec = bg.evaluate_gate_by_segment("duration", "m", "b", y, m, b, c, seg, "null", n_boot=2000)
        false += any(v["status"] == bg.PROMOTED for v in rec["segments"].values())
    assert false <= 3   # nominal family-wise rate 2.5% -> expect about 0.5 of 20


def test_small_segment_inherits_global_verdict():
    y, m, b, c, seg = _data()
    seg = seg.copy()
    seg[:10] = "sp_tiny"
    rec = bg.evaluate_gate_by_segment("duration", "m", "b", y, m, b, c, seg, "synthetic",
                                      global_record={"status": bg.DEMOTED})
    tiny = rec["segments"]["sp_tiny"]
    assert not tiny["tested"] and tiny["status"] == bg.DEMOTED and tiny["verdict_source"] == "inherited_global"


def test_verdict_lookup():
    seg_rec = {"segments": {"sp_14_plus": {"status": bg.PROMOTED, "verdict_source": "segment_test_holm"},
                            "sp_3_5": {"status": bg.DEMOTED, "verdict_source": "segment_test_holm"}},
               "global_status": bg.DEMOTED}
    g = {"status": bg.DEMOTED}
    assert bg.verdict_for("duration", g, seg_rec, 20)["status"] == bg.PROMOTED
    assert bg.verdict_for("duration", g, seg_rec, 3)["status"] == bg.DEMOTED
    assert bg.verdict_for("duration", g, seg_rec, 1)["source"] == "inherited_global_unseen_segment"
    assert bg.verdict_for("duration", g, None, 20) == {"status": bg.DEMOTED, "segment": None, "source": "global"}


def test_forecaster_serves_per_segment():
    pytest.importorskip("ngboost")
    import pickle
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from app.ai import forecasting_engine as fe
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(1, 21, 200), rng.integers(0, 6, 200), rng.uniform(1, 5, 200)]).astype(float)
    m = NGBRegressor(Dist=Normal, n_estimators=30, verbose=False, random_state=0).fit(X, X[:, 0] * 0.8 + 1)
    with open(fe.NGB_MODEL_PATH, "wb") as f:
        pickle.dump(m, f)
    meta = {"status": "trained", "model_version": "t", "training_median_days": 2.0,
            "gate": {"status": bg.DEMOTED, "model": "NGBoost", "baseline": "training_median"},
            "segment_gate": {"global_status": bg.DEMOTED, "segments": {
                "sp_14_plus": {"status": bg.PROMOTED, "verdict_source": "segment_test_holm"},
                "sp_3_5": {"status": bg.DEMOTED, "verdict_source": "segment_test_holm"}}}}
    with open(fe.NGB_META_PATH, "w") as f:
        json.dump(meta, f)
    big, small = fe.predict(20, "Backend"), fe.predict(3, "Backend")
    assert big["point_source"] == "model" and big["gate_verdict"]["segment"] == "sp_14_plus"
    assert small["point_source"] == "baseline_training_median"
    os.remove(fe.NGB_META_PATH)
