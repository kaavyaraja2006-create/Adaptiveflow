"""Gate-variant ablation on the released out-of-fold files, plus 1,000-run null simulations."""
import json
import sys
import time

import numpy as np
import pandas as pd

BACKEND, RES, OUT = sys.argv[1:4]
sys.path.insert(0, BACKEND)
from app.ai import baseline_gate as bg  # noqa: E402


def naive_ci(em, eb, n_boot=2000, seed=0):
    return bg.clustered_paired_ci(em, eb, np.arange(len(em)), n_boot=n_boot, seed=seed)


def variants(y, pm, pb, cl, seg=None):
    em, eb = y - pm, y - pb
    d, lo, hi, k = bg.clustered_paired_ci(em, eb, cl)
    dn, lon, hin, _ = naive_ci(em, eb)
    out = {"delta": d, "clustered": (lo, hi), "naive": (lon, hin),
           "promote_clustered": hi < 0, "promote_naive": hin < 0, "promote_threshold": d < 0,
           "width_ratio": (hi - lo) / (hin - lon)}
    return out


def seg_variants(y, pm, pb, cl, segs):
    res = {}
    pv_c, pv_n = {}, {}
    for s in sorted(set(segs)):
        m = segs == s
        if m.sum() < bg.MIN_SEGMENT_ROWS or len(np.unique(cl[m])) < bg.MIN_SEGMENT_CLUSTERS:
            continue
        em, eb = y[m] - pm[m], y[m] - pb[m]
        _, _, _, _, p_c = bg._clustered_boot(em, eb, cl[m], 4000, 0, 0.95)
        _, _, _, _, p_n = bg._clustered_boot(em, eb, np.arange(m.sum()), 4000, 0, 0.95)
        pv_c[s], pv_n[s] = p_c, p_n
        res[s] = {"delta": float(np.mean(np.abs(em)) - np.mean(np.abs(eb))), "p_clustered": p_c, "p_naive": p_n}
    holm_c, holm_n = bg.holm(pv_c), bg.holm(pv_n)
    for s in res:
        res[s].update(threshold=res[s]["delta"] < 0, naive_unadj=pv_n[s] <= 0.025, clustered_unadj=pv_c[s] <= 0.025,
                      clustered_holm=holm_c[s], naive_holm=holm_n[s])
    return res


d = pd.read_csv(f"{RES}/table1_duration_oof.csv")
r = pd.read_csv(f"{RES}/table2_recovery_risk_oof.csv")
P = "Persistence (today's risk)"
out = {"global": [], "segments": {}}
for task, df, base, models, key in [
        ("duration", d, "Training median", ["Ridge", "Random Forest", "Gradient Boosting", "NGBoost (AdaptiveFlow)"], "story_points"),
        ("risk", r, P, ["Ridge regression", "Random Forest", "Gradient Boosting", "BiGRU (AdaptiveFlow)"], P)]:
    y, pb, cl = df.y.values, df[base].values, df.cluster.values
    segs = df[key].map(bg.SEGMENT_FUNCTIONS[task]).values
    for m in models:
        v = variants(y, df[m].values, pb, cl)
        out["global"].append({"task": task, "model": m, **{k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()}})
        out["segments"][f"{task}|{m}"] = seg_variants(y, df[m].values, pb, cl, segs)

# ------------------------------------------------------------------ null simulations
def sim(seed, n_clusters=60, per=40, tau=1.0):
    """Model and baseline are equally good on average, but the advantage varies by cluster (sd tau):
    rows within a cluster share the cluster's advantage, which is what a naive bootstrap ignores."""
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_clusters), per)
    sp = rng.choice([1, 3, 8, 20], len(cl))
    y = 10 + rng.normal(0, 3, n_clusters)[cl] + rng.normal(0, 2, len(cl))
    shift = rng.normal(0, tau, n_clusters)
    shift -= shift.mean()                          # exactly no average advantage
    base = y + rng.normal(0, 2.0, len(y))
    model = y + rng.normal(0, 1.0, len(y)) * np.exp(shift[cl] * 0.25)   # cluster-varying error scale
    # recentre so that the model's MAE equals the baseline's MAE in expectation
    return y, model, base, cl, np.array([bg.duration_segment(s) for s in sp])


def calibrate(seed):
    y, m, b, cl, segs = sim(seed)
    em, eb = np.abs(y - m), np.abs(y - b)
    # rescale model errors so pooled MAEs are equal (null holds in the population of clusters)
    m2 = y - (y - m) * (eb.mean() / em.mean())
    return y, m2, b, cl, segs


N = 1000
t0 = time.time()
cnt = {"global_clustered": 0, "global_naive": 0, "global_threshold": 0,
       "seg_clustered_holm": 0, "seg_clustered_unadj": 0, "seg_naive_unadj": 0, "seg_threshold": 0}
for s in range(N):
    y, m, b, cl, segs = calibrate(10_000 + s)
    v = variants(y, m, b, cl)
    cnt["global_clustered"] += v["promote_clustered"]
    cnt["global_naive"] += v["promote_naive"]
    cnt["global_threshold"] += v["promote_threshold"]
    sv = seg_variants(y, m, b, cl, segs)
    cnt["seg_clustered_holm"] += any(x["clustered_holm"] for x in sv.values())
    cnt["seg_clustered_unadj"] += any(x["clustered_unadj"] for x in sv.values())
    cnt["seg_naive_unadj"] += any(x["naive_unadj"] for x in sv.values())
    cnt["seg_threshold"] += any(x["threshold"] for x in sv.values())
out["null"] = {"runs": N, "rates": {k: v / N for k, v in cnt.items()}, "seconds": time.time() - t0,
               "design": "60 clusters x 40 rows; equal pooled MAE; cluster-varying model advantage (exp(0.25*N(0,1)) "
                         "scale on model errors), rescaled so the pooled MAE of model and baseline are equal"}
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(json.dumps(out["null"], indent=1))
for g in out["global"]:
    print(g["task"], g["model"][:14], round(g["delta"], 4), [round(x, 3) for x in g["clustered"]],
          [round(x, 3) for x in g["naive"]], g["promote_clustered"], g["promote_naive"], g["promote_threshold"],
          round(g["width_ratio"], 2))
for k, v in out["segments"].items():
    print(k, {s: (round(x["delta"], 3), x["threshold"], x["naive_unadj"], x["clustered_unadj"], x["clustered_holm"]) for s, x in v.items()})
