"""Segment gate, cross-fitted policies and naive-vs-clustered interval width for the absolute-error (L1)
duration model of evaluate_extended.py (rich features, projects held out); also per-fold velocity deltas.

    python analysis/extended_segments.py results
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from app.ai import baseline_gate as bg  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "results"
M, BASE = "Gradient Boosting (L1)", "Training median"
d = pd.read_csv(os.path.join(R, "itemlet_duration_rich_projects_oof.csv"))
seg = d.story_points.map(bg.duration_segment).values
y, base, cl, pm = d.y.values, d[BASE].values, d.cluster.values, d[M].values
out = {}
rec = bg.evaluate_gate_by_segment("duration", M, BASE, y, pm, base, cl, seg, "itemlet rich")
out["segments"] = {s: {k: r.get(k) for k in ("n_rows", "n_clusters", "delta", "ci_low", "ci_high", "p_one_sided",
                                             "status")} for s, r in rec["segments"].items()}
sg, ss = np.empty(len(y)), np.empty(len(y))
for k in sorted(d.fold.unique()):
    te = (d.fold == k).values
    tr = ~te
    g = bg.evaluate_gate("duration", M, BASE, y[tr], pm[tr], base[tr], cl[tr], "cf")
    sr = bg.evaluate_gate_by_segment("duration", M, BASE, y[tr], pm[tr], base[tr], cl[tr], seg[tr], "cf",
                                     global_record=g)
    sg[te] = pm[te] if g["status"] == bg.PROMOTED else base[te]
    use = np.array([sr["segments"].get(s, {}).get("status", sr["global_status"]) == bg.PROMOTED for s in seg[te]])
    ss[te] = np.where(use, pm[te], base[te])
mae = lambda p: float(np.mean(np.abs(y - p)))
out["policy_mae"] = {"baseline": mae(base), "always_model": mae(pm), "global_gate": mae(sg), "segment_gate": mae(ss)}
_, lo, hi, _, _ = bg._clustered_boot(np.abs(y - pm), np.abs(y - base), cl, 2000, 0, 0.95)
_, lo2, hi2, _, _ = bg._clustered_boot(np.abs(y - pm), np.abs(y - base), np.arange(len(y)), 2000, 0, 0.95)
out["ci_width"] = {"clustered": hi - lo, "row_bootstrap": hi2 - lo2}
out["per_fold_delta"] = {int(k): float(np.mean(np.abs(g.y - g[M]) - np.abs(g.y - g[BASE]))) for k, g in d.groupby("fold")}
v = pd.read_csv(os.path.join(R, "itemlet_velocity_oof.csv"))
out["velocity_per_fold_delta"] = {c: [float(np.mean(np.abs(g.y - g[c]) - np.abs(g.y - g["Trailing 7-day mean"])))
                                      for _, g in v.groupby("fold")] for c in (M, "BiGRU (AdaptiveFlow)")}
json.dump(out, open(os.path.join(R, "extended_segments.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
