"""1,000-run null simulations (population null with cluster-level heterogeneity) and cross-fitted
serving error of naive / threshold segment gates on the real out-of-fold files."""
import json
import sys
import time

import numpy as np
import pandas as pd

BACKEND, RES, OUT = sys.argv[1:4]
sys.path.insert(0, BACKEND)
from app.ai import baseline_gate as bg  # noqa: E402

SEGS = np.array(["sp_1_2", "sp_3_5", "sp_6_13", "sp_14_plus"])


def sim(seed, n_clusters=60, per=40, tau=1.0, sigma=2.0):
    """Population null: in every cluster and segment the model's advantage u ~ N(0, tau) has mean zero,
    so on average neither is better; rows within a cluster share u (what a naive bootstrap ignores)."""
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_clusters), per)
    seg_idx = rng.integers(0, 4, len(cl))
    u = rng.normal(0, tau, (n_clusters, 4))[cl, seg_idx]
    eb = 6 + np.abs(rng.normal(0, sigma, len(cl)))
    em = eb + u + rng.normal(0, sigma, len(cl))
    return np.zeros(len(cl)), em, eb, cl, SEGS[seg_idx]   # |y - p| = p with y = 0


def p_value(em, eb, cl, naive):
    clusters = np.arange(len(em)) if naive else cl
    return bg._clustered_boot(em, eb, clusters, 2000, 0, 0.95)


N = 1000
t0 = time.time()
cnt = dict.fromkeys(["global_clustered", "global_naive", "global_threshold", "seg_clustered_holm",
                     "seg_clustered_unadj", "seg_naive_unadj", "seg_threshold"], 0)
for s in range(N):
    y, pm, pb, cl, segs = sim(20_000 + s)
    d, lo, hi, _, _ = p_value(pm, pb, cl, False)
    dn, lon, hin, _, _ = p_value(pm, pb, cl, True)
    cnt["global_clustered"] += hi < 0
    cnt["global_naive"] += hin < 0
    cnt["global_threshold"] += d < 0
    pc, pn, th = {}, {}, {}
    for g in SEGS:
        m = segs == g
        dc, _, _, _, pc[g] = p_value(pm[m], pb[m], cl[m], False)
        _, _, _, _, pn[g] = p_value(pm[m], pb[m], cl[m], True)
        th[g] = dc < 0
    cnt["seg_clustered_holm"] += any(bg.holm(pc).values())
    cnt["seg_clustered_unadj"] += any(v <= 0.025 for v in pc.values())
    cnt["seg_naive_unadj"] += any(v <= 0.025 for v in pn.values())
    cnt["seg_threshold"] += any(th.values())
null = {"runs": N, "rates": {k: v / N for k, v in cnt.items()}, "seconds": round(time.time() - t0, 1),
        "design": "60 clusters x 40 rows, 4 segments; per-cluster-segment model advantage ~ N(0, 1) with "
                  "population mean 0; row noise sd 2; 2,000 bootstrap draws"}

# ------------------------------------------------------------------ cross-fitted serving with naive gates (real data)
def crossfit(df, base, model, key, comp, rule):
    y, pm, pb, cl = df.y.values, df[model].values, df[base].values, df.cluster.values
    seg = df[key].map(bg.SEGMENT_FUNCTIONS[comp]).values
    served = pb.copy()
    for k in sorted(df.fold.unique()):
        te = (df.fold == k).values
        tr = ~te
        pvals, deltas = {}, {}
        for g in sorted(set(seg[tr])):
            m = tr & (seg == g)
            if m.sum() < bg.MIN_SEGMENT_ROWS:
                continue
            clusters = np.arange(m.sum()) if rule == "naive_unadj" else cl[m]
            dlt, _, _, _, p = bg._clustered_boot(y[m] - pm[m], y[m] - pb[m], clusters, 2000, 0, 0.95)
            pvals[g], deltas[g] = p, dlt
        if rule == "threshold":
            prom = {g for g in deltas if deltas[g] < 0}
        elif rule == "naive_unadj":
            prom = {g for g in pvals if pvals[g] <= 0.025}
        else:
            prom = {g for g, ok in bg.holm(pvals).items() if ok}
        use = te & np.isin(seg, list(prom))
        served[use] = pm[use]
    return float(np.mean(np.abs(y - served))), float(np.mean(np.isclose(served, pm) & ~np.isclose(pm, pb)))


d = pd.read_csv(f"{RES}/table1_duration_oof.csv")
r = pd.read_csv(f"{RES}/table2_recovery_risk_oof.csv")
P = "Persistence (today's risk)"
cf = []
for comp, df, base, key, models in [
        ("duration", d, "Training median", "story_points", ["Ridge", "Random Forest", "Gradient Boosting", "NGBoost (AdaptiveFlow)"]),
        ("risk", r, P, P, ["Gradient Boosting"])]:
    for m in models:
        row = {"task": comp, "model": m, "baseline_mae": float(np.mean(np.abs(df.y - df[base])))}
        for rule in ("threshold", "naive_unadj", "clustered_holm"):
            row[rule] = crossfit(df, base, m, key, comp, rule)
        cf.append(row)
json.dump({"null": null, "crossfit_naive": cf}, open(OUT, "w"), indent=1, default=float)
print(json.dumps(null, indent=1))
for row in cf:
    print(row["task"], row["model"][:12], round(row["baseline_mae"], 4),
          {k: (round(v[0], 4), round(v[1], 3)) for k, v in row.items() if isinstance(v, tuple)})
