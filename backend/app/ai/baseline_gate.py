"""
Baseline-gated learning contract (paper Section IV-M).

A learned component may drive a downstream decision only if it beats a
named trivial baseline on held-out data. Concretely, with per-row absolute
errors e_m (model) and e_b (baseline) on the same held-out rows:

    delta = mean(|e_m|) - mean(|e_b|)          (negative = model is better)

and a 95% confidence interval for delta from a clustered (paired) bootstrap
that resamples whole clusters (project, employee or grouping unit), so rows
from the same cluster are never treated as independent.

    promoted  <=>  upper bound of the 95% CI < 0

Otherwise the component is demoted: the baseline's point estimate is served
downstream, and both values plus the interval are stored with the output.
A component with no defined baseline, no gate record, or fewer than two
held-out clusters cannot be promoted.

The gate decides only WHICH POINT ESTIMATE IS SERVED. A demoted model's other
outputs (e.g. NGBoost's predictive interval) remain available, labelled as
coming from a demoted model.
"""
import json
import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_store")
GATE_VERSION = "baseline-gate-v1.0"
N_BOOT = 2000
CONFIDENCE = 0.95

PROMOTED = "gated_promoted"
DEMOTED = "gated_demoted"


def clustered_paired_ci(err_model: np.ndarray, err_base: np.ndarray, clusters: Sequence,
                        n_boot: int = N_BOOT, seed: int = 0, confidence: float = CONFIDENCE):
    """Point estimate and percentile CI of mean(|e_m|) - mean(|e_b|), resampling whole clusters."""
    delta, lo, hi, k, _ = _clustered_boot(err_model, err_base, clusters, n_boot, seed, confidence)
    return delta, lo, hi, k


def _clustered_boot(err_model, err_base, clusters, n_boot, seed, confidence):
    d = np.abs(np.asarray(err_model, float)) - np.abs(np.asarray(err_base, float))
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    sums = np.bincount(inv, weights=d, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    # one-sided bootstrap p-value for H0: delta >= 0 (model no better than baseline)
    p = (int(np.sum(boot >= 0)) + 1) / (n_boot + 1)
    return delta, float(lo), float(hi), k, float(p)


def evaluate_gate(component: str, model_name: str, baseline_name: Optional[str],
                  y_true, pred_model, pred_base, clusters, dataset: str,
                  n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Run the gate on paired held-out predictions and return a storable record."""
    y_true = np.asarray(y_true, float)
    pred_model = np.asarray(pred_model, float)
    record = {
        "gate_version": GATE_VERSION,
        "component": component,
        "model": model_name,
        "baseline": baseline_name,
        "dataset": dataset,
        "metric": "MAE",
        "n_rows": int(len(y_true)),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "rule": "promote iff upper bound of the clustered 95% CI of MAE(model) - MAE(baseline) < 0",
    }
    if baseline_name is None or pred_base is None:
        return {**record, "status": DEMOTED, "reason": "no_defined_baseline",
                "model_mae": float(np.mean(np.abs(y_true - pred_model))) if len(y_true) else None,
                "baseline_mae": None, "delta": None, "ci_low": None, "ci_high": None, "n_clusters": 0}
    pred_base = np.asarray(pred_base, float)
    e_m, e_b = y_true - pred_model, y_true - pred_base
    model_mae, base_mae = float(np.mean(np.abs(e_m))), float(np.mean(np.abs(e_b)))
    n_clusters = int(len(np.unique(np.asarray(clusters))))
    if n_clusters < 2:
        return {**record, "status": DEMOTED, "reason": "fewer_than_two_held_out_clusters",
                "model_mae": model_mae, "baseline_mae": base_mae,
                "delta": model_mae - base_mae, "ci_low": None, "ci_high": None, "n_clusters": n_clusters}
    delta, lo, hi, k, p = _clustered_boot(e_m, e_b, clusters, n_boot, seed, CONFIDENCE)
    promoted = hi < 0
    return {**record, "status": PROMOTED if promoted else DEMOTED,
            "reason": "model_beats_baseline" if promoted else "model_does_not_beat_baseline",
            "model_mae": round(model_mae, 6), "baseline_mae": round(base_mae, 6),
            "delta": round(delta, 6), "ci_low": round(lo, 6), "ci_high": round(hi, 6),
            "p_one_sided": round(p, 6), "n_clusters": k, "n_boot": n_boot, "seed": seed}


def missing_gate(component: str, model_name: str) -> dict:
    """What callers get when a model exists but was never gated (e.g. trained by an older script)."""
    return {"gate_version": GATE_VERSION, "component": component, "model": model_name,
            "status": DEMOTED, "reason": "no_gate_record"}


def is_promoted(record: Optional[dict]) -> bool:
    return bool(record) and record.get("status") == PROMOTED


def summary(record: Optional[dict]) -> Optional[dict]:
    """Compact version of a gate record for attaching to every prediction."""
    if not record:
        return None
    keys = ("status", "reason", "model", "baseline", "dataset", "model_mae", "baseline_mae",
            "delta", "ci_low", "ci_high", "n_clusters")
    return {k: record.get(k) for k in keys}


def gate_path(component: str) -> str:
    return os.path.join(MODEL_DIR, f"gate_{component}.json")


def save_gate(record: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(gate_path(record["component"]), "w") as f:
        json.dump(record, f, indent=2)


def save_segment_gate(record: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"gate_{record['component']}_segments.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)


# ====================================================================== segment-wise gate
# A single verdict per component hides heterogeneity: a model can lose to the baseline overall
# yet win on, say, large tickets. The segment-wise gate runs the same paired, cluster-bootstrap
# test inside pre-declared segments and promotes a segment only if the Holm-corrected one-sided
# test (H0: model is no better than the baseline) rejects at alpha = 0.025 — the same level as
# the global rule "upper bound of the 95% CI < 0". Segments are defined only from information
# available at prediction time and are fixed in code BEFORE any evaluation (no post-hoc search).
# A segment with too little held-out evidence is not tested and inherits the global verdict.

SEGMENT_ALPHA = 0.025
SEGMENT_N_BOOT = 4000
MIN_SEGMENT_ROWS = 30
MIN_SEGMENT_CLUSTERS = 5

SEGMENTATIONS = {
    # component -> (name, description); the functions below implement them
    "duration": ("story_point_band", "story points requested: 1-2 / 3-5 / 6-13 / 14+"),
    "risk": ("current_risk_band", "today's deterministic RecoveryRisk band: low / moderate / elevated / severe"),
}


def duration_segment(story_points) -> str:
    sp = float(story_points or 3)
    if sp <= 2:
        return "sp_1_2"
    if sp <= 5:
        return "sp_3_5"
    if sp <= 13:
        return "sp_6_13"
    return "sp_14_plus"


def risk_segment(current_risk) -> str:
    r = float(current_risk or 0.0)
    if r >= 0.75:
        return "severe"
    if r >= 0.55:
        return "elevated"
    if r >= 0.30:
        return "moderate"
    return "low"


SEGMENT_FUNCTIONS = {"duration": duration_segment, "risk": risk_segment}


def holm(pvalues: dict, alpha: float = SEGMENT_ALPHA) -> dict:
    """Holm step-down: returns {key: rejected?}. Controls the family-wise error rate at alpha."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(order)
    out = {k: False for k in pvalues}
    for i, k in enumerate(order):
        if pvalues[k] <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def evaluate_gate_by_segment(component: str, model_name: str, baseline_name: str, y_true, pred_model, pred_base,
                             clusters, segments, dataset: str, global_record: Optional[dict] = None,
                             n_boot: int = SEGMENT_N_BOOT, seed: int = 0) -> dict:
    y_true, pred_model, pred_base = (np.asarray(a, float) for a in (y_true, pred_model, pred_base))
    clusters, segments = np.asarray(clusters), np.asarray(segments)
    if global_record is None:
        global_record = evaluate_gate(component, model_name, baseline_name, y_true, pred_model, pred_base,
                                      clusters, dataset)
    seg_records, pvals = {}, {}
    for seg in sorted(set(segments.tolist())):
        m = segments == seg
        n_rows, n_cl = int(m.sum()), int(len(np.unique(clusters[m])))
        if n_rows < MIN_SEGMENT_ROWS or n_cl < MIN_SEGMENT_CLUSTERS:
            seg_records[seg] = {"segment": seg, "tested": False, "n_rows": n_rows, "n_clusters": n_cl,
                                "reason": "insufficient_segment_evidence",
                                "model_mae": float(np.mean(np.abs(y_true[m] - pred_model[m]))) if n_rows else None,
                                "baseline_mae": float(np.mean(np.abs(y_true[m] - pred_base[m]))) if n_rows else None}
            continue
        r = evaluate_gate(component, model_name, baseline_name, y_true[m], pred_model[m], pred_base[m],
                          clusters[m], f"{dataset} [segment {seg}]", n_boot=n_boot, seed=seed)
        seg_records[seg] = {"segment": seg, "tested": True, **{k: r.get(k) for k in (
            "n_rows", "n_clusters", "model_mae", "baseline_mae", "delta", "ci_low", "ci_high", "p_one_sided")}}
        pvals[seg] = r["p_one_sided"]
    rejected = holm(pvals)
    g_status = global_record.get("status", DEMOTED)
    for seg, rec in seg_records.items():
        if rec["tested"]:
            rec["status"] = PROMOTED if rejected[seg] else DEMOTED
            rec["verdict_source"] = "segment_test_holm"
        else:
            rec["status"] = g_status
            rec["verdict_source"] = "inherited_global"
    name, desc = SEGMENTATIONS[component]
    return {"gate_version": GATE_VERSION, "component": component, "model": model_name, "baseline": baseline_name,
            "dataset": dataset, "segmentation": name, "segmentation_definition": desc,
            "multiple_testing": f"Holm, one-sided alpha={SEGMENT_ALPHA}, {len(pvals)} tested segments",
            "min_segment_rows": MIN_SEGMENT_ROWS, "min_segment_clusters": MIN_SEGMENT_CLUSTERS,
            "n_boot": n_boot, "seed": seed, "global_status": g_status, "segments": seg_records,
            "evaluated_at": datetime.utcnow().isoformat() + "Z"}


def verdict_for(component: str, global_record: Optional[dict], segment_record: Optional[dict], segment_key) -> dict:
    """Which verdict applies to one prediction. Uses the segment verdict when a segment-wise record
    exists for this component; otherwise the global verdict. Unknown segments inherit the global verdict."""
    g_status = (global_record or {}).get("status", DEMOTED)
    if segment_record:
        seg = SEGMENT_FUNCTIONS[component](segment_key)
        rec = (segment_record.get("segments") or {}).get(seg)
        if rec is not None:
            return {"status": rec["status"], "segment": seg, "source": rec["verdict_source"]}
        return {"status": g_status, "segment": seg, "source": "inherited_global_unseen_segment"}
    return {"status": g_status, "segment": None, "source": "global"}


def segment_policy_mae(y_true, pred_model, pred_base, segments, seg_record: dict) -> float:
    """MAE when each row is served by its segment's verdict (model if promoted, else baseline)."""
    y_true, pred_model, pred_base = (np.asarray(a, float) for a in (y_true, pred_model, pred_base))
    segments = np.asarray(segments)
    use_model = np.array([seg_record["segments"].get(s, {}).get("status", seg_record["global_status"]) == PROMOTED
                          for s in segments])
    served = np.where(use_model, pred_model, pred_base)
    return float(np.mean(np.abs(y_true - served)))
