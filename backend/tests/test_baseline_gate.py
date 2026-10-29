"""Baseline gate (paper Section IV-M)."""
import json
import os

import numpy as np

from app.ai import baseline_gate as bg


def _data(n_clusters=40, per=30, model_noise=1.0, base_noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_clusters), per)
    shift = rng.normal(0, 2, n_clusters)[clusters]          # correlated within cluster
    y = 10 + shift + rng.normal(0, 1, len(clusters))
    pm = y + rng.normal(0, model_noise, len(y))
    pb = y + rng.normal(0, base_noise, len(y))
    return y, pm, pb, clusters


def test_clearly_better_model_is_promoted():
    y, pm, pb, c = _data(model_noise=0.5, base_noise=3.0)
    r = bg.evaluate_gate("t", "m", "b", y, pm, pb, c, dataset="synthetic")
    assert r["status"] == bg.PROMOTED and r["ci_high"] < 0


def test_equal_model_is_demoted():
    y, pm, pb, c = _data(model_noise=1.0, base_noise=1.0)
    r = bg.evaluate_gate("t", "m", "b", y, pm, pb, c, dataset="synthetic")
    assert r["status"] == bg.DEMOTED and r["ci_low"] < 0 < r["ci_high"]


def test_worse_model_is_demoted():
    y, pm, pb, c = _data(model_noise=3.0, base_noise=0.5)
    r = bg.evaluate_gate("t", "m", "b", y, pm, pb, c, dataset="synthetic")
    assert r["status"] == bg.DEMOTED and r["ci_low"] > 0


def test_no_baseline_cannot_be_promoted():
    y, pm, _, c = _data(model_noise=0.01)
    r = bg.evaluate_gate("t", "m", None, y, pm, None, c, dataset="synthetic")
    assert r["status"] == bg.DEMOTED and r["reason"] == "no_defined_baseline"


def test_single_cluster_cannot_be_promoted():
    y, pm, pb, _ = _data(model_noise=0.1, base_noise=5)
    r = bg.evaluate_gate("t", "m", "b", y, pm, pb, np.zeros(len(y)), dataset="synthetic")
    assert r["status"] == bg.DEMOTED and r["reason"] == "fewer_than_two_held_out_clusters"


def test_missing_record_is_demoted():
    assert not bg.is_promoted(None)
    assert not bg.is_promoted(bg.missing_gate("risk", "BiGRU"))


def test_clustered_ci_wider_than_naive():
    """Resampling clusters must not give a narrower interval than resampling rows."""
    y, pm, pb, c = _data(n_clusters=20, per=50, model_noise=1.0, base_noise=1.2)
    _, lo_c, hi_c, _ = bg.clustered_paired_ci(y - pm, y - pb, c)
    _, lo_r, hi_r, _ = bg.clustered_paired_ci(y - pm, y - pb, np.arange(len(y)))
    assert (hi_c - lo_c) >= (hi_r - lo_r)


def test_deterministic_given_seed():
    y, pm, pb, c = _data()
    a = bg.evaluate_gate("t", "m", "b", y, pm, pb, c, dataset="s")
    b = bg.evaluate_gate("t", "m", "b", y, pm, pb, c, dataset="s")
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_demoted_duration_serves_training_median(tmp_path):
    """forecasting_engine.predict must serve the training median when NGBoost is demoted."""
    pytest = __import__("pytest")
    pytest.importorskip("ngboost")
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    import pickle
    from app.ai import forecasting_engine as fe
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(1, 13, 200), rng.integers(0, 6, 200), rng.uniform(1, 5, 200)]).astype(float)
    y = X[:, 0] * 0.8 + rng.normal(0, 1, 200)
    m = NGBRegressor(Dist=Normal, n_estimators=30, verbose=False, random_state=0).fit(X, np.maximum(0.25, y))
    with open(fe.NGB_MODEL_PATH, "wb") as f:
        pickle.dump(m, f)
    meta = {"status": "trained", "model_version": "test", "training_median_days": 2.0,
            "gate": {"status": bg.DEMOTED, "reason": "model_does_not_beat_baseline", "model": "NGBoost",
                     "baseline": "training_median"}}
    with open(fe.NGB_META_PATH, "w") as f:
        json.dump(meta, f)
    out = fe.predict(8, "Backend", 3.2)
    assert out["point_source"] == "baseline_training_median"
    assert out["p50_hours"] == round(2.0 * fe.EFFECTIVE_HOURS_PER_DAY, 1)
    assert out["p80_hours"] >= out["p50_hours"] and out["p95_hours"] >= out["p80_hours"]
    assert out["gate"]["status"] == bg.DEMOTED and "learned_point_hours" in out
    # promoted -> model's own point forecast
    meta["gate"]["status"] = bg.PROMOTED
    with open(fe.NGB_META_PATH, "w") as f:
        json.dump(meta, f)
    assert fe.predict(8, "Backend", 3.2)["point_source"] == "model"
    os.remove(fe.NGB_META_PATH)
