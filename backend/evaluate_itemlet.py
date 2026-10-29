"""
Paper-style model comparison tables on the real Itemlet dataset (mean +/- std over k cross-validation folds).

Run AFTER you have the data in place (same as train_itemlet.py):
    python evaluate_itemlet.py                      # both tables
    python evaluate_itemlet.py --folds 5 --max-rows 200000

Table 1 - Duration forecasting (how long an issue takes), 5-fold CV with whole PROJECTS held out:
    models: Empirical baseline, Ridge, Random Forest, Gradient Boosting, NGBoost (the model AdaptiveFlow uses)
    regression: MAE, RMSE, R2      classification ("slow issue" = longer than the training 75th percentile):
    Precision, Recall, F1, Accuracy, ROC-AUC      plus 80% interval coverage (probabilistic models) and latency.
Table 2 - Recovery-risk forecasting (next-day risk from a 14-day window), 5-fold CV with whole DEVELOPERS held out:
    models: Predict-mean, Persistence, Ridge/Logistic, Random Forest, Gradient Boosting, BiGRU (the model AdaptiveFlow uses)
    regression: MAE, RMSE, R2      classification ("elevated risk" = next-day risk >= 0.55, the app's band): P/R/F1/Acc/AUC.

Everything is computed from the real data; nothing is hard-coded. Outputs go to ./results/.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

import itemlet_training as it
from app.ai import bigru

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


# ------------------------------------------------------------------ metrics helpers
def cls_metrics(y_true, score, pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    out = {"Precision": precision_score(y_true, pred, zero_division=0), "Recall": recall_score(y_true, pred, zero_division=0),
           "F1": f1_score(y_true, pred, zero_division=0), "Accuracy": accuracy_score(y_true, pred)}
    out["ROC-AUC"] = roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan
    return out


def reg_metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "R2": r2_score(y, p)}


def latency_ms(fn, X, n=150):
    """mean milliseconds to score ONE sample (what the API does per request)."""
    idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    fn(X[idx[:3]][:1])  # warm-up
    t0 = time.perf_counter()
    for i in idx:
        fn(X[i:i + 1])
    return (time.perf_counter() - t0) / len(idx) * 1000.0


def summarize(rows: dict, order):
    """rows[model] = list of per-fold metric dicts -> DataFrame of 'mean +/- std' strings."""
    table, raw = [], []
    for m in order:
        if m not in rows or not rows[m]:
            continue
        d = pd.DataFrame(rows[m])
        rec = {"Model": m}
        for c in d.columns:
            mu, sd = np.nanmean(d[c]), np.nanstd(d[c])
            rec[c] = f"{mu:.3f} ± {sd:.3f}" if c != "Latency (ms)" else f"{mu:.3f} ± {sd:.3f}"
        table.append(rec)
        raw.append({"Model": m, **{c + " (mean)": np.nanmean(d[c]) for c in d.columns},
                    **{c + " (std)": np.nanstd(d[c]) for c in d.columns}})
    return pd.DataFrame(table), pd.DataFrame(raw)


# ------------------------------------------------------------------ Table 1: duration
def duration_frame(df, max_rows):
    d = it.clean_completed(df)
    sp = pd.to_numeric(d["story_points"], errors="coerce")
    d = d[sp.notna() & (sp > 0) & (sp <= 100)].copy()
    d["story_points"] = np.maximum(1, np.round(sp[d.index])).astype(int)
    d["task_type"] = it.classify_task_type(d)
    d = d[d["actual_days"] <= min(np.percentile(d["actual_days"], 99), 120.0)]
    d["actual_days"] = d["actual_days"].clip(lower=0.25)
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    return d.reset_index(drop=True)


def eval_duration(df, folds, ngb_samples, ngb_estimators, max_rows):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold
    d = duration_frame(df, max_rows)
    it.log(f"[Duration] {len(d):,} issues, {d['project'].nunique() if 'project' in d else 'n/a'} projects")
    X = np.column_stack([d["story_points"].values.astype(float), d["task_type"].map(it._type_index).values.astype(float),
                         d["story_points"].values / 2.5])
    y = d["actual_days"].values.astype(float)
    tcode = d["task_type"].map(it._type_index).values
    if "project" in d.columns and d["project"].nunique() >= folds:
        splits = list(GroupKFold(n_splits=folds).split(X, y, d["project"]))
        split_kind = "held-out projects"
    else:
        splits = list(KFold(folds, shuffle=True, random_state=0).split(X)); split_kind = "random rows"
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        have_ngb = True
    except ImportError:
        have_ngb = False
        it.log("  ngboost not installed - NGBoost row will be skipped")
    rows = {}
    groups = d["project"].values if "project" in d.columns else np.arange(len(d))
    oof_parts = []
    for k, (tr, va) in enumerate(splits, 1):
        thr = np.percentile(y[tr], 75)
        ytrue_c = (y[va] > thr).astype(int)
        fold_oof = {"fold": np.full(len(va), k), "row": va, "cluster": groups[va], "y": y[va],
                    "story_points": X[va, 0], "task_type_code": tcode[va]}

        def add(name, pred, cov=np.nan, fn=None):
            r = {**reg_metrics(y[va], pred), **cls_metrics(ytrue_c, pred, (pred > thr).astype(int)),
                 "80% Coverage": cov, "Latency (ms)": latency_ms(fn, X[va]) if fn else np.nan}
            rows.setdefault(name, []).append(r)
            fold_oof[name] = np.asarray(pred, float)

        # trivial baseline for the gate: constant training median (optimal constant under absolute error)
        med = float(np.median(y[tr]))
        add("Training median", np.full(len(va), med), fn=lambda x: med)

        # empirical per-type baseline: predicts mean ratio * story points; interval from p10/p90 ratios
        ratio = y[tr] / np.maximum(1, X[tr, 0])
        mean_r = {t: (ratio[tcode[tr] == t].mean() if (tcode[tr] == t).sum() > 5 else ratio.mean()) for t in range(7)}
        p10 = {t: np.percentile(ratio[tcode[tr] == t] if (tcode[tr] == t).sum() > 5 else ratio, 10) for t in range(7)}
        p90 = {t: np.percentile(ratio[tcode[tr] == t] if (tcode[tr] == t).sum() > 5 else ratio, 90) for t in range(7)}
        base_pred = np.array([mean_r[t] * s for t, s in zip(tcode[va], X[va, 0])])
        lo = np.array([p10[t] * s for t, s in zip(tcode[va], X[va, 0])]); hi = np.array([p90[t] * s for t, s in zip(tcode[va], X[va, 0])])
        add("Empirical baseline", base_pred, float(np.mean((y[va] >= lo) & (y[va] <= hi))),
            lambda x: mean_r[int(x[0, 1])] * x[0, 0])
        ridge = Ridge(alpha=1.0).fit(X[tr], y[tr]); add("Ridge", ridge.predict(X[va]), fn=ridge.predict)
        rf = RandomForestRegressor(100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[tr][:60000], y[tr][:60000])
        add("Random Forest", rf.predict(X[va]), fn=rf.predict)
        gb = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[tr], y[tr]); add("Gradient Boosting", gb.predict(X[va]), fn=gb.predict)
        if have_ngb:
            sel = np.random.default_rng(k).choice(tr, min(ngb_samples, len(tr)), replace=False)
            ngb = NGBRegressor(Dist=Normal, n_estimators=ngb_estimators, verbose=False, random_state=0).fit(X[sel], y[sel])
            dist = ngb.pred_dist(X[va])
            lo10, hi90 = dist.dist.ppf(0.10), dist.dist.ppf(0.90)
            cov = float(np.mean((y[va] >= lo10) & (y[va] <= hi90)))
            add("NGBoost (AdaptiveFlow)", ngb.predict(X[va]), cov, ngb.predict)
            fold_oof["NGBoost q10"], fold_oof["NGBoost q90"] = lo10, hi90
        fold_oof["Empirical q10"], fold_oof["Empirical q90"] = lo, hi
        oof_parts.append(pd.DataFrame(fold_oof))
        it.log(f"  fold {k}/{folds} done")
    order = ["Training median", "Empirical baseline", "Ridge", "Random Forest", "Gradient Boosting", "NGBoost (AdaptiveFlow)"]
    t, raw = summarize(rows, order)
    note = (f"{folds}-fold CV, {split_kind}; n={len(d):,} issues. 'Slow issue' = longer than the training-fold 75th percentile of "
            "duration (score = predicted days). MAE/RMSE in days. Coverage = share of outcomes inside the model's 10-90% interval.")
    return t, raw, note, pd.concat(oof_parts, ignore_index=True)


# ------------------------------------------------------------------ Table 2: BiGRU
def summary_features(W):
    """(N,14,10) window -> 40 summary features for the classical baselines."""
    return np.concatenate([W[:, -1, :], W.mean(1), W.max(1), W.std(1)], axis=1)


def fit_bigru(Xtr, ytr_r, ytr_v, epochs, batch, lr, patience, seed):
    import copy
    rng = np.random.default_rng(seed)
    n_in = max(1, int(0.1 * len(Xtr)))
    perm = rng.permutation(len(Xtr)); iv, itr = perm[:n_in], perm[n_in:]
    vc = float(np.percentile(ytr_v, 99)) or 1.0
    mean = Xtr[itr].reshape(-1, bigru.INPUT_SIZE).mean(0); std = Xtr[itr].reshape(-1, bigru.INPUT_SIZE).std(0) + 1e-6
    Xn = (Xtr - mean) / std
    yv = np.minimum(ytr_v, vc)
    model, best, best_s, bad = bigru.BiGRUWorkloadModel(), None, np.inf, 0
    for ep in range(epochs):
        p = rng.permutation(itr)
        for i in range(0, len(p), batch):
            b = p[i:i + batch]
            model.train_step(Xn[b], ytr_r[b], yv[b], lr)
        r, v = model.predict(Xn[iv])
        s = float(np.mean((r - ytr_r[iv]) ** 2)) + 0.05 * float(np.mean((v - yv[iv]) ** 2))
        if s < best_s - 1e-5:
            best, best_s, bad = copy.deepcopy(model), s, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best or model, mean, std


def eval_bigru(df, folds, args):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    seqs = it.build_developer_sequences(df, args.min_issues, args.max_devs, args.max_days, args.active_cap_days)
    if len(seqs) < folds * 2:
        raise SystemExit(f"Only {len(seqs)} developers - not enough for {folds}-fold CV.")
    kf = KFold(folds, shuffle=True, random_state=0)
    rows, pos_rate, thr_used = {}, [], None
    oof_parts = []
    cap_each = args.max_windows
    for k, (tr_i, va_i) in enumerate(kf.split(np.arange(len(seqs))), 1):
        rng = np.random.default_rng(k)
        Xtr, rtr, vtr, ctr = it.make_windows([seqs[i] for i in tr_i], args.stride)
        Xva, rva, vva, cva, gva = it.make_windows([seqs[i] for i in va_i], args.stride, return_groups=True)
        if len(Xtr) > cap_each:
            s = rng.choice(len(Xtr), cap_each, replace=False); Xtr, rtr, vtr, ctr = Xtr[s], rtr[s], vtr[s], ctr[s]
        if len(Xva) > cap_each // 3:
            s = rng.choice(len(Xva), cap_each // 3, replace=False); Xva, rva, vva, cva, gva = Xva[s], rva[s], vva[s], cva[s], gva[s]
        thr = args.risk_threshold
        if thr is None:
            thr = 0.55 if (rtr >= 0.55).mean() >= 0.05 else float(np.percentile(rtr, 80))
        thr_used = thr
        yc = (rva >= thr).astype(int)
        pos_rate.append(float(yc.mean()))
        fold_oof = {"fold": np.full(len(rva), k), "cluster": gva, "y": rva}

        def add(name, score, fn=None):
            pred_c = (score >= thr).astype(int)
            rows.setdefault(name, []).append({**reg_metrics(rva, score), **cls_metrics(yc, score, pred_c),
                                              "Latency (ms)": latency_ms(fn, Xva) if fn else np.nan})
            fold_oof[name] = np.asarray(score, float)

        mu = rtr.mean()
        add("Predict-mean", np.full(len(rva), mu), lambda x: mu)
        add("Persistence (today's risk)", cva, lambda x: float(x[0, -1, 6]))
        Ftr, Fva = summary_features(Xtr), summary_features(Xva)
        sc = StandardScaler().fit(Ftr); Ftr_s, Fva_s = sc.transform(Ftr), sc.transform(Fva)
        ridge = Ridge(1.0).fit(Ftr_s, rtr)
        add("Ridge regression", np.clip(ridge.predict(Fva_s), 0, 1), lambda x: ridge.predict(sc.transform(summary_features(x))))
        sub = rng.choice(len(Ftr), min(40000, len(Ftr)), replace=False)
        rf = RandomForestRegressor(100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Ftr[sub], rtr[sub])
        add("Random Forest", np.clip(rf.predict(Fva), 0, 1), lambda x: rf.predict(summary_features(x)))
        gb = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Ftr, rtr)
        add("Gradient Boosting", np.clip(gb.predict(Fva), 0, 1), lambda x: gb.predict(summary_features(x)))
        model, mean, std = fit_bigru(Xtr, rtr, vtr, args.epochs, args.batch, args.lr, args.patience, k)
        pr = model.predict((Xva - mean) / std)[0]
        add("BiGRU (AdaptiveFlow)", pr, lambda x: model.predict((x - mean) / std))
        oof_parts.append(pd.DataFrame(fold_oof))
        it.log(f"  fold {k}/{folds} done (positive rate {yc.mean():.1%}, threshold {thr:.2f})")
    order = ["Predict-mean", "Persistence (today's risk)", "Ridge regression", "Random Forest", "Gradient Boosting", "BiGRU (AdaptiveFlow)"]
    t, raw = summarize(rows, order)
    note = (f"{folds}-fold CV, whole developers held out; {len(seqs)} developers. Task: predict next-day recovery risk from a 14-day window. "
            f"'Elevated risk' = next-day risk >= {thr_used:.2f} (mean positive rate {np.mean(pos_rate):.1%}). Regression metrics are on the 0-1 risk score.")
    return t, raw, note, pd.concat(oof_parts, ignore_index=True)


# ------------------------------------------------------------------ baseline gate (paper Section IV-M)
def gate_from_oof(oof: pd.DataFrame, component: str, baseline: str, models: list, dataset: str):
    """Apply the baseline gate to every learned model's pooled out-of-fold predictions."""
    from app.ai import baseline_gate
    recs = []
    for m in models:
        if m not in oof.columns:
            continue
        r = baseline_gate.evaluate_gate(component=component, model_name=m, baseline_name=baseline,
                                        y_true=oof["y"].values, pred_model=oof[m].values,
                                        pred_base=oof[baseline].values, clusters=oof["cluster"].values,
                                        dataset=dataset)
        recs.append(r)
    os.makedirs(OUT, exist_ok=True)
    import json
    with open(os.path.join(OUT, f"gate_{component}.json"), "w") as f:
        json.dump(recs, f, indent=2, default=str)
    tab = pd.DataFrame([{k: r.get(k) for k in ("model", "baseline", "model_mae", "baseline_mae", "delta",
                                                   "ci_low", "ci_high", "n_clusters", "status")} for r in recs])
    tab.to_csv(os.path.join(OUT, f"gate_{component}.csv"), index=False)
    print("\n" + _md(tab))
    return recs


def segment_gate_from_oof(oof: pd.DataFrame, component: str, baseline: str, models: list, key_col: str,
                          dataset: str):
    """Segment-wise gate (novelty #1) on pooled out-of-fold predictions, plus a CROSS-FITTED policy
    comparison: for each fold k, verdicts are decided on the other folds only and applied to fold k,
    so the reported policy errors are not inflated by choosing segments on the evaluation data."""
    from app.ai import baseline_gate as bg
    import json
    seg_fn = bg.SEGMENT_FUNCTIONS[component]
    seg = oof[key_col].map(seg_fn).values
    y, base = oof["y"].values, oof[baseline].values
    detail_rows, policy_rows, records = [], [], {}
    for m in models:
        if m not in oof.columns:
            continue
        pm = oof[m].values
        rec = bg.evaluate_gate_by_segment(component, m, baseline, y, pm, base, oof["cluster"].values, seg, dataset)
        records[m] = rec
        for s, r in rec["segments"].items():
            detail_rows.append({"model": m, "segment": s, **{k: r.get(k) for k in (
                "n_rows", "n_clusters", "model_mae", "baseline_mae", "delta", "ci_low", "ci_high",
                "p_one_sided", "status", "verdict_source")}})
        # cross-fitted policies
        served_global, served_seg = np.empty(len(y)), np.empty(len(y))
        n_seg_promoted = []
        for k in sorted(oof["fold"].unique()):
            te = (oof["fold"] == k).values
            tr = ~te
            g = bg.evaluate_gate(component, m, baseline, y[tr], pm[tr], base[tr], oof["cluster"].values[tr], dataset)
            sr = bg.evaluate_gate_by_segment(component, m, baseline, y[tr], pm[tr], base[tr],
                                             oof["cluster"].values[tr], seg[tr], dataset, global_record=g)
            n_seg_promoted.append(sum(r["status"] == bg.PROMOTED for r in sr["segments"].values()))
            served_global[te] = pm[te] if g["status"] == bg.PROMOTED else base[te]
            use = np.array([sr["segments"].get(s, {}).get("status", sr["global_status"]) == bg.PROMOTED
                            for s in seg[te]])
            served_seg[te] = np.where(use, pm[te], base[te])
        mae = lambda p: float(np.mean(np.abs(y - p)))
        cl = oof["cluster"].values
        d_seg_vs_base = bg.clustered_paired_ci(y - served_seg, y - base, cl)
        policy_rows.append({"model": m, "always_baseline_mae": mae(base), "always_model_mae": mae(pm),
                            "global_gate_mae (cross-fitted)": mae(served_global),
                            "segment_gate_mae (cross-fitted)": mae(served_seg),
                            "segment_vs_baseline_delta": d_seg_vs_base[0],
                            "ci_low": d_seg_vs_base[1], "ci_high": d_seg_vs_base[2],
                            "share_rows_served_by_model": float(np.mean(served_seg == pm)),
                            "segments_promoted_per_fold": "/".join(map(str, n_seg_promoted))})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"gate_{component}_segments.json"), "w") as f:
        json.dump(records, f, indent=2, default=str)
    pd.DataFrame(detail_rows).to_csv(os.path.join(OUT, f"gate_{component}_segments.csv"), index=False)
    pol = pd.DataFrame(policy_rows)
    pol.to_csv(os.path.join(OUT, f"gate_{component}_policy.csv"), index=False)
    print("\n" + _md(pd.DataFrame(detail_rows).round(4)))
    print("\n" + _md(pol.round(4)))
    return records, pol


# ------------------------------------------------------------------ output
def save_outputs(name, table, raw, note, title):
    os.makedirs(OUT, exist_ok=True)
    raw.to_csv(os.path.join(OUT, f"{name}_raw.csv"), index=False)
    table.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    md = f"### {title}\n\n" + table.to_markdown(index=False) + f"\n\n_{note}_\n" if _has_tabulate() else \
        f"### {title}\n\n" + _md(table) + f"\n\n_{note}_\n"
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        _png(table, title, os.path.join(OUT, f"{name}.png"))
    except Exception as e:
        it.log(f"(PNG skipped: {e})")
    print("\n" + md)


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


def _md(t):
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _png(table, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table.fillna("-").replace("nan ± nan", "-")
    fig_w = max(10, 1.6 * len(t.columns)); fig, ax = plt.subplots(figsize=(fig_w, 0.55 * (len(t) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=t.values, colLabels=list(t.columns), loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#999999")
        if r == 0:
            cell.set_facecolor("#222222"); cell.set_text_props(color="white", weight="bold")
        elif "AdaptiveFlow" in str(t.iloc[r - 1, 0]):
            cell.set_facecolor("#e8f0fe")
    ax.set_title(title, fontsize=11, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    default_dir = os.path.join(os.path.dirname(HERE), "data", "itemlet")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(default_dir, "itemlet_dataset.parquet"))
    ap.add_argument("--dictionary", default=os.path.join(default_dir, "data_dictionary.csv"))
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--only", choices=["all", "duration", "bigru"], default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample the dataset")
    ap.add_argument("--duration-rows", type=int, default=150000)
    ap.add_argument("--ngb-samples", type=int, default=20000)
    ap.add_argument("--ngb-estimators", type=int, default=100)
    ap.add_argument("--risk-threshold", type=float, default=None, help="default: 0.55, or the 80th percentile if fewer than 5%% of windows reach it")
    ap.add_argument("--min-issues", type=int, default=40); ap.add_argument("--max-devs", type=int, default=2000)
    ap.add_argument("--max-days", type=int, default=540); ap.add_argument("--active-cap-days", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--max-windows", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    need = {"all": sorted(set(it.REQUIRED_DURATION + it.REQUIRED_BIGRU)), "duration": it.REQUIRED_DURATION, "bigru": it.REQUIRED_BIGRU}[a.only]
    df, _ = it.load_dataset(a.data, dict(m.split("=", 1) for m in a.map), a.dictionary, a.max_rows, need)
    if a.only in ("all", "duration"):
        t, raw, note, oof = eval_duration(df, a.folds, a.ngb_samples, a.ngb_estimators, a.duration_rows)
        save_outputs("table1_duration", t, raw, note, "Table 1. Issue-duration forecasting on Itemlet (mean ± std over folds)")
        oof.to_csv(os.path.join(OUT, "table1_duration_oof.csv"), index=False)
        gate_from_oof(oof, "duration", "Training median",
                      ["Ridge", "Random Forest", "Gradient Boosting", "NGBoost (AdaptiveFlow)"],
                      "D2-itemlet (5-fold, projects held out)")
        segment_gate_from_oof(oof, "duration", "Training median",
                              ["Ridge", "Random Forest", "Gradient Boosting", "NGBoost (AdaptiveFlow)"],
                              "story_points", "D2-itemlet (5-fold, projects held out)")
    if a.only in ("all", "bigru"):
        t, raw, note, oof = eval_bigru(df, a.folds, a)
        save_outputs("table2_recovery_risk", t, raw, note, "Table 2. Next-day recovery-risk forecasting on Itemlet (mean ± std over folds)")
        oof.to_csv(os.path.join(OUT, "table2_recovery_risk_oof.csv"), index=False)
        gate_from_oof(oof, "risk", "Persistence (today's risk)",
                      ["Ridge regression", "Random Forest", "Gradient Boosting", "BiGRU (AdaptiveFlow)"],
                      "D3-itemlet (5-fold, grouping units held out)")
        segment_gate_from_oof(oof, "risk", "Persistence (today's risk)",
                              ["Ridge regression", "Random Forest", "Gradient Boosting", "BiGRU (AdaptiveFlow)"],
                              "Persistence (today's risk)", "D3-itemlet (5-fold, grouping units held out)")
    it.log(f"Saved tables (.md / .csv / .png) in {OUT}")


if __name__ == "__main__":
    main()
