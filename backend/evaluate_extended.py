"""
Extended evaluation for the revised paper (run after evaluate_itemlet.py, same data arguments):

  E1b  Duration with richer, leakage-safe features
       - story points, keyword task type (as before)
       - priority, Jira issue type, number of components and labels
       - title text (hashed bag of words reduced by truncated SVD, fitted on training rows only)
       - assignee and project history: mean log-duration and count of issues RESOLVED BEFORE this issue was
         CREATED (no look-ahead)
       evaluated twice: (a) 5-fold, whole projects held out; (b) chronological: train on issues created before the
       80th percentile of creation time AND resolved before it, test on issues created after it.

  E2b  Next-day completed story points (velocity): an OBSERVED outcome (story points of issues resolved that day),
       not a formula of the inputs. Same 14-day windows and unit-held-out folds as E2. Named baseline: the trailing
       7-day mean (what the app serves when no model is promoted); persistence also reported.

Every task writes per-row out-of-fold predictions, a results table, and global (and, for duration, segment-wise)
gate verdicts, exactly like evaluate_itemlet.py.

    python evaluate_extended.py                       # all parts
    python evaluate_extended.py --only duration       # E1b only
    python evaluate_extended.py --only velocity       # E2b only
    python evaluate_extended.py --data <file.csv> --tag tawos   # another Jira export (e.g. TAWOS), see README
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

import evaluate_itemlet as ev
import itemlet_training as it
from app.ai import bigru

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = ev.OUT
PRIORITY_RANK = {"blocker": 5, "highest": 5, "critical": 4, "high": 4, "major": 3, "medium": 3, "normal": 3,
                 "minor": 2, "low": 2, "trivial": 1, "lowest": 1}


# ------------------------------------------------------------------ feature engineering
def _prior_stats(d: pd.DataFrame, key: str):
    """Mean log(1+days) and count of issues in the same group RESOLVED strictly before each issue's CREATION."""
    if key not in d.columns:
        return np.zeros(len(d)), np.zeros(len(d))
    g = d[key].astype(str).fillna("?")
    ev_tab = pd.DataFrame({"g": g, "t": d["resolved"], "v": np.log1p(d["actual_days"])}).sort_values(["g", "t"])
    ev_tab["cs"] = ev_tab.groupby("g")["v"].cumsum()
    ev_tab["cn"] = ev_tab.groupby("g").cumcount() + 1
    q = pd.DataFrame({"g": g, "t": d["created"] - pd.Timedelta(microseconds=1), "i": np.arange(len(d))}).sort_values("t")
    m = pd.merge_asof(q, ev_tab.sort_values("t")[["g", "t", "cs", "cn"]], on="t", by="g", direction="backward")
    m = m.sort_values("i")
    cn = m["cn"].fillna(0).values
    mean = np.where(cn > 0, m["cs"].fillna(0).values / np.maximum(cn, 1), np.nan)
    return mean, cn


def _count_list(s: pd.Series) -> np.ndarray:
    def cnt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        if isinstance(x, (list, tuple, np.ndarray)):
            return len(x)
        x = str(x).strip("[] ")
        return 0 if not x or x.lower() in ("nan", "none") else len([p for p in x.replace(";", ",").split(",") if p.strip()])
    return np.array([cnt(x) for x in s])


def rich_frame(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    d = ev.duration_frame(df, max_rows)
    d["tcode"] = d["task_type"].map(it._type_index)
    d["prio"] = d["priority"].astype(str).str.lower().map(PRIORITY_RANK).fillna(3) if "priority" in d else 3
    if "issue_type" in d:
        top = d["issue_type"].astype(str).str.lower().value_counts().index[:12]
        d["jtype"] = d["issue_type"].astype(str).str.lower().map({t: i for i, t in enumerate(top)}).fillna(len(top))
    else:
        d["jtype"] = 0
    d["n_comp"] = _count_list(d["components"]) if "components" in d else 0
    d["n_label"] = _count_list(d["labels"]) if "labels" in d else 0
    d["title_len"] = d["summary"].astype(str).str.len() if "summary" in d else 0
    d["created_hour"] = d["created"].dt.hour
    d["created_dow"] = d["created"].dt.weekday
    d["asg_mean"], d["asg_n"] = _prior_stats(d, "assignee")
    d["prj_mean"], d["prj_n"] = _prior_stats(d, "project")
    return d


NUM_COLS = ["story_points", "tcode", "prio", "jtype", "n_comp", "n_label", "title_len", "created_hour",
            "created_dow", "asg_mean", "asg_n", "prj_mean", "prj_n"]


def text_features(train_text, test_text, k=32, seed=0):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=2 ** 16, alternate_sign=False, norm="l2", ngram_range=(1, 2),
                           stop_words="english")
    a, b = hv.transform(train_text), hv.transform(test_text)
    svd = TruncatedSVD(k, random_state=seed).fit(a)
    return svd.transform(a), svd.transform(b)


def design(d, tr, va, with_text=True):
    Xn = d[NUM_COLS].astype(float).values
    med = np.nanmedian(Xn[tr], axis=0)
    Xn = np.where(np.isnan(Xn), med, Xn)
    if with_text and "summary" in d:
        ta, tb = text_features(d["summary"].astype(str).values[tr], d["summary"].astype(str).values[va])
        return np.hstack([Xn[tr], ta]), np.hstack([Xn[va], tb])
    return Xn[tr], Xn[va]


# ------------------------------------------------------------------ E1b duration, rich features
def run_duration(d, splits, label, ngb_samples, ngb_estimators):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        have_ngb = True
    except ImportError:
        have_ngb = False
    y = d["actual_days"].values.astype(float)
    groups = d["project"].values if "project" in d else np.arange(len(d))
    rows, parts = {}, []
    for k, (tr, va) in enumerate(splits, 1):
        t0 = time.time()
        Xtr, Xva = design(d, tr, va)
        fo = {"fold": np.full(len(va), k), "row": va, "cluster": groups[va], "y": y[va],
              "story_points": d["story_points"].values[va]}
        thr = np.percentile(y[tr], 75)

        def add(name, pred, cov=np.nan):
            r = {**ev.reg_metrics(y[va], pred), **ev.cls_metrics((y[va] > thr).astype(int), pred,
                                                                  (pred > thr).astype(int)), "80% Coverage": cov}
            rows.setdefault(name, []).append(r)
            fo[name] = np.asarray(pred, float)

        add("Training median", np.full(len(va), float(np.median(y[tr]))))
        sc = StandardScaler().fit(Xtr)
        add("Ridge", Ridge(1.0).fit(sc.transform(Xtr), y[tr]).predict(sc.transform(Xva)))
        add("Gradient Boosting", HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, y[tr]).predict(Xva))
        gbl = HistGradientBoostingRegressor(max_iter=300, loss="absolute_error", random_state=0).fit(Xtr, y[tr])
        add("Gradient Boosting (L1)", gbl.predict(Xva))
        if have_ngb:
            sel = np.random.default_rng(k).choice(len(Xtr), min(ngb_samples, len(Xtr)), replace=False)
            ngb = NGBRegressor(Dist=Normal, n_estimators=ngb_estimators, verbose=False, random_state=0)
            ngb.fit(Xtr[sel], y[tr][sel])
            dist = ngb.pred_dist(Xva)
            lo, hi = dist.dist.ppf(0.10), dist.dist.ppf(0.90)
            add("NGBoost (AdaptiveFlow)", ngb.predict(Xva), float(np.mean((y[va] >= lo) & (y[va] <= hi))))
        parts.append(pd.DataFrame(fo))
        it.log(f"  [{label}] split {k}/{len(splits)} done ({time.time() - t0:.0f}s, {len(tr):,} train / {len(va):,} test)")
    order = ["Training median", "Ridge", "Gradient Boosting", "Gradient Boosting (L1)", "NGBoost (AdaptiveFlow)"]
    t, raw = ev.summarize(rows, [o for o in order if o in rows])
    return t, raw, pd.concat(parts, ignore_index=True)


def chrono_split(d, q=0.8):
    cut = d["created"].quantile(q)
    tr = np.where((d["created"] < cut) & (d["resolved"] < cut))[0]
    va = np.where(d["created"] >= cut)[0]
    return [(tr, va)], cut


# ------------------------------------------------------------------ E2b velocity (observed outcome)
def run_velocity(df, folds, a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    seqs = it.build_developer_sequences(df, a.min_issues, a.max_devs, a.max_days, a.active_cap_days)
    kf = KFold(folds, shuffle=True, random_state=0)
    rows, parts = {}, []
    for k, (tr_i, va_i) in enumerate(kf.split(np.arange(len(seqs))), 1):
        rng = np.random.default_rng(k)
        Xtr, rtr, vtr, _ = it.make_windows([seqs[i] for i in tr_i], a.stride)
        Xva, rva, vva, _, gva = it.make_windows([seqs[i] for i in va_i], a.stride, return_groups=True)
        if len(Xtr) > a.max_windows:
            s = rng.choice(len(Xtr), a.max_windows, replace=False); Xtr, rtr, vtr = Xtr[s], rtr[s], vtr[s]
        if len(Xva) > a.max_windows // 3:
            s = rng.choice(len(Xva), a.max_windows // 3, replace=False); Xva, vva, gva = Xva[s], vva[s], gva[s]
        fo = {"fold": np.full(len(vva), k), "cluster": gva, "y": vva}

        def add(name, pred):
            rows.setdefault(name, []).append(ev.reg_metrics(vva, pred))
            fo[name] = np.asarray(pred, float)

        add("Trailing 7-day mean", Xva[:, -7:, 0].mean(1))
        add("Persistence (today)", Xva[:, -1, 0])
        add("Predict-mean", np.full(len(vva), float(vtr.mean())))
        Ftr, Fva = ev.summary_features(Xtr), ev.summary_features(Xva)
        sc = StandardScaler().fit(Ftr)
        add("Ridge regression", np.maximum(0, Ridge(1.0).fit(sc.transform(Ftr), vtr).predict(sc.transform(Fva))))
        add("Gradient Boosting", np.maximum(0, HistGradientBoostingRegressor(max_iter=200, random_state=0)
                                            .fit(Ftr, vtr).predict(Fva)))
        add("Gradient Boosting (L1)", np.maximum(0, HistGradientBoostingRegressor(max_iter=200, loss="absolute_error",
                                                                                  random_state=0).fit(Ftr, vtr).predict(Fva)))
        model, mean, std = ev.fit_bigru(Xtr, rtr, vtr, a.epochs, a.batch, a.lr, a.patience, k)
        add("BiGRU (AdaptiveFlow)", np.maximum(0, model.predict((Xva - mean) / std)[1]))
        parts.append(pd.DataFrame(fo))
        it.log(f"  [velocity] fold {k}/{folds} done ({len(vva):,} windows)")
    order = ["Predict-mean", "Persistence (today)", "Trailing 7-day mean", "Ridge regression", "Gradient Boosting",
             "Gradient Boosting (L1)", "BiGRU (AdaptiveFlow)"]
    t, raw = ev.summarize(rows, order)
    return t, raw, pd.concat(parts, ignore_index=True)


def main():
    default_dir = os.path.join(os.path.dirname(HERE), "data", "itemlet")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(default_dir, "itemlet_dataset.parquet"))
    ap.add_argument("--dictionary", default=os.path.join(default_dir, "data_dictionary.csv"))
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--tag", default="itemlet", help="prefix for output files (e.g. tawos)")
    ap.add_argument("--only", choices=["all", "duration", "velocity"], default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--duration-rows", type=int, default=150000)
    ap.add_argument("--ngb-samples", type=int, default=20000)
    ap.add_argument("--ngb-estimators", type=int, default=100)
    ap.add_argument("--min-issues", type=int, default=40); ap.add_argument("--max-devs", type=int, default=2000)
    ap.add_argument("--max-days", type=int, default=540); ap.add_argument("--active-cap-days", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--max-windows", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    need = sorted(set(it.REQUIRED_DURATION + (it.REQUIRED_BIGRU if a.only != "duration" else [])))
    overrides = dict(m.split("=", 1) for m in a.map)
    df, mapping = it.load_dataset(a.data, overrides, a.dictionary, a.max_rows, need)
    it.log("Rich-feature columns found: " + ", ".join(f for f in ("summary", "priority", "issue_type", "components",
                                                                    "labels", "project", "assignee") if f in df.columns))
    tag = a.tag
    if a.only in ("all", "duration"):
        from sklearn.model_selection import GroupKFold
        d = rich_frame(df, a.duration_rows)
        it.log(f"[E1b] {len(d):,} issues; features: {NUM_COLS} + 32 title-text components")
        splits = list(GroupKFold(n_splits=a.folds).split(d, groups=d["project"])) if "project" in d else None
        t, raw, oof = run_duration(d, splits, "projects held out", a.ngb_samples, a.ngb_estimators)
        name = f"{tag}_duration_rich_projects"
        ev.save_outputs(name, t, raw, "Rich features; 5-fold, whole projects held out.", f"Duration, rich features ({tag}), projects held out")
        oof.to_csv(os.path.join(OUT, f"{name}_oof.csv"), index=False)
        models = [m for m in ["Ridge", "Gradient Boosting", "Gradient Boosting (L1)", "NGBoost (AdaptiveFlow)"] if m in oof]
        ev.gate_from_oof(oof, f"{tag}_duration_rich_projects", "Training median", models, f"{tag} rich, projects held out")
        chrono, cut = chrono_split(d)
        t, raw, oof = run_duration(d, chrono, f"chronological, cut {cut.date()}", a.ngb_samples, a.ngb_estimators)
        name = f"{tag}_duration_rich_chrono"
        ev.save_outputs(name, t, raw, f"Rich features; train created and resolved before {cut.date()}, test created after.",
                        f"Duration, rich features ({tag}), chronological split")
        oof.to_csv(os.path.join(OUT, f"{name}_oof.csv"), index=False)
        ev.gate_from_oof(oof, f"{tag}_duration_rich_chrono", "Training median", models, f"{tag} rich, chronological")
    if a.only in ("all", "velocity"):
        t, raw, oof = run_velocity(df, a.folds, a)
        name = f"{tag}_velocity"
        ev.save_outputs(name, t, raw, "Next-day completed story points; 5-fold, grouping units held out.",
                        f"Next-day velocity ({tag})")
        oof.to_csv(os.path.join(OUT, f"{name}_oof.csv"), index=False)
        ev.gate_from_oof(oof, f"{tag}_velocity", "Trailing 7-day mean",
                         ["Persistence (today)", "Ridge regression", "Gradient Boosting", "Gradient Boosting (L1)",
                          "BiGRU (AdaptiveFlow)"], f"{tag} velocity, units held out")
    it.log(f"Saved results in {OUT}")


if __name__ == "__main__":
    main()
