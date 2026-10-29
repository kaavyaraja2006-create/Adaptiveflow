"""
Train AdaptiveFlow's two learned models on the real Itemlet dataset
(727k Jira issues, 204 open-source projects; https://zenodo.org/records/19411554).

NOTHING in the existing app is modified. This module only:
  * reads itemlet_dataset.parquet (or .csv),
  * trains the SAME model classes the app already ships
        - duration forecaster   (ai/forecasting_engine.py, NGBoost)  -> model_store/ngboost_duration.pkl,
                                                                         ngboost_duration_meta.json, duration_baseline.json
        - BiGRU workload model  (ai/bigru.py, NumPy BiGRU)           -> model_store/bigru_weights.npz, bigru_meta.json
  * writes the artifacts in exactly the formats the app already loads, so the API, the Digital Twin /
    Monte Carlo simulator and the allocation engine pick them up with no code change.

Dependency-graph (IssueLink), allocation scoring and Monte Carlo have no learnable parameters in this
project - Monte Carlo and allocation simply consume the duration + workload models trained here.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `app.ai.*` imports work from any cwd

from app.ai import bigru  # numpy only  # noqa: E402

MODEL_DIR = bigru.MODEL_DIR
# ---- constants mirrored from ai/forecasting_engine.py (imported when possible) --------------------------
TASK_TYPES = ["Backend", "Frontend", "QA", "DevOps", "Mobile", "Security"]
NGB_MODEL_PATH = os.path.join(MODEL_DIR, "ngboost_duration.pkl")
NGB_META_PATH = os.path.join(MODEL_DIR, "ngboost_duration_meta.json")
BASELINE_PATH = os.path.join(MODEL_DIR, "duration_baseline.json")
DUR_MIN_SAMPLES = 30
try:  # keep in lock-step with the app if its dependencies (sqlalchemy) are installed
    from app.ai import forecasting_engine as _fe
    TASK_TYPES, NGB_MODEL_PATH, NGB_META_PATH, BASELINE_PATH = (
        _fe.TASK_TYPES, _fe.NGB_MODEL_PATH, _fe.NGB_META_PATH, _fe.BASELINE_PATH)
    DUR_MIN_SAMPLES = _fe.MIN_TRAINING_SAMPLES
except Exception:
    pass

# RecoveryRisk weights from ai/workload_engine.py
W_LOAD, W_HIGH, W_AFTER, W_CTX, W_REST = 0.35, 0.20, 0.20, 0.15, 0.25
try:
    from app.ai import workload_engine as _we
    W_LOAD, W_HIGH, W_AFTER, W_CTX, W_REST = (
        _we.W_LOAD_RATIO, _we.W_HIGH_COMPLEXITY, _we.W_AFTER_HOURS, _we.W_CONTEXT_SWITCH, _we.W_REST_DAYS)
except Exception:
    pass

SPRINT_CAPACITY = 40.0   # same "points per sprint" constant the workload engine uses


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================================================
# 1. Column resolution (Itemlet has 108 columns; names are auto-detected, and can be overridden)
# =====================================================================================================
LOGICAL_FIELDS: Dict[str, List[str]] = {
    "story_points": ["storypoints", "storypoint", "storypointeffort", "sp"],
    "created":      ["created", "createddate", "createdat", "creationdate", "createdtime", "createdts"],
    "resolved":     ["resolutiondate", "resolved", "resolveddate", "resolvedat", "resolutiontime", "closeddate"],
    "assignee":     ["assignee", "assigneeid", "assigneename", "assigneekey", "assigneedisplayname", "assigneeusername"],
    "issue_type":   ["issuetype", "issuetypename", "type"],
    "project":      ["project", "projectkey", "projectid", "projectname"],
    "summary":      ["summary", "title"],
    "components":   ["components", "component"],
    "labels":       ["labels", "label"],
    "priority":     ["priority", "priorityname"],
}
REQUIRED_DURATION = ["story_points", "created", "resolved"]
REQUIRED_BIGRU = ["created", "resolved", "assignee"]
_BAD_TOKENS = ("count", "ratio", "days", "hours", "avg", "mean", "num", "flag", "diff", "delta", "duration",
               "isnull", "has", "log", "norm", "bin", "bucket", "score", "rate", "total", "per", "std",
               "update", "sprint", "first", "last", "reporter", "creator")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_columns(columns: List[str], overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    out: Dict[str, Optional[str]] = {}
    for field, cands in LOGICAL_FIELDS.items():
        if field in overrides:
            if overrides[field] not in columns:
                raise SystemExit(f"--map {field}={overrides[field]}: no such column in the dataset")
            out[field] = overrides[field]
            continue
        hit = next((by_norm[c] for c in cands if c in by_norm), None)
        if hit is None:  # fuzzy: candidate is a substring of the column name, skipping derived-looking columns
            fuzzy = [c for c in columns if any(k in _norm(c) for k in cands[:2])
                     and not any(b in _norm(c).replace(cands[0], "", 1) for b in _BAD_TOKENS)]
            hit = min(fuzzy, key=len) if fuzzy else None
        out[field] = hit
    return out


def print_column_report(df_cols: List[str], mapping: Dict[str, Optional[str]], dict_path: Optional[str]):
    descr = {}
    if dict_path and os.path.exists(dict_path):
        try:
            dd = pd.read_csv(dict_path)
            name_col = next((c for c in dd.columns if _norm(c) in ("column", "columnname", "variable",
                             "variablename", "name", "field", "feature")), dd.columns[0])
            desc_col = next((c for c in dd.columns if "desc" in c.lower() or "defin" in c.lower()), None)
            for _, r in dd.iterrows():
                descr[str(r[name_col])] = str(r[desc_col]) if desc_col else ""
        except Exception as e:
            log(f"(could not parse data dictionary: {e})")
    log(f"Dataset has {len(df_cols)} columns. Column mapping used:")
    for field, col in mapping.items():
        extra = f"  - {descr[col][:110]}" if col in descr else ""
        print(f"    {field:<13} -> {col}{extra}")


# =====================================================================================================
# 2. Loading + cleaning
# =====================================================================================================
def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        med = s.dropna().median() if s.notna().any() else 0
        unit = "ms" if med > 1e11 else "s"
        out = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        out = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return out.dt.tz_convert("UTC").dt.tz_localize(None)


def load_dataset(path: str, overrides: Dict[str, str], dict_path: Optional[str], max_rows: Optional[int],
                 need: List[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    if not os.path.exists(path):
        raise SystemExit(f"Dataset not found: {path}\nDownload itemlet_dataset.parquet from "
                         "https://zenodo.org/records/19411554 into data/itemlet/")
    is_parquet = path.lower().endswith(".parquet")
    if is_parquet:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pyarrow is required to read Parquet:  pip install -r requirements-itemlet.txt")
        columns = pq.ParquetFile(path).schema.names
    else:
        columns = list(pd.read_csv(path, nrows=0).columns)
    mapping = resolve_columns(columns, overrides)
    print_column_report(columns, mapping, dict_path)
    missing = [f for f in need if not mapping.get(f)]
    if missing:
        raise SystemExit(
            f"\nCould not find column(s) for: {missing}.\nRun with --inspect to list every column, find the right "
            f"names in data_dictionary.csv, then pass e.g.  --map {missing[0]}=<column_name>")
    use = sorted({c for c in mapping.values() if c})
    log(f"Reading {len(use)} of {len(columns)} columns from {os.path.basename(path)} ...")
    df = pd.read_parquet(path, columns=use) if is_parquet else pd.read_csv(path, usecols=use)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=0)
    df = df.rename(columns={c: f for f, c in mapping.items() if c})
    log(f"Loaded {len(df):,} issues.")
    return df, mapping


def _text_blob(df: pd.DataFrame) -> pd.Series:
    parts = []
    for f in ("issue_type", "components", "labels", "summary"):
        if f in df.columns:
            parts.append(df[f].astype(str).str.lower().fillna(""))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    blob = parts[0]
    for p in parts[1:]:
        blob = blob + " " + p
    return blob


# keyword -> AdaptiveFlow skill category (first match wins, ordered most specific first)
_TYPE_PATTERNS = [
    ("Security", r"secur|vulnerab|\bcve\b|xss|csrf|inject|exploit|encrypt|ssl|tls|authenticat|authoriz|permission|privilege"),
    ("Mobile",   r"android|\bios\b|iphone|ipad|mobile|apk|swift|kotlin|react[- ]native"),
    ("QA",       r"\btests?\b|testing|unit test|junit|flaky|coverage|regression|\bqa\b|integration test|test case"),
    ("DevOps",   r"build|deploy|\bci\b|jenkins|docker|kubernetes|k8s|pipeline|infra|release|maven|gradle|packag|install|config|ansible|helm"),
    ("Frontend", r"\bui\b|\bux\b|css|html|browser|front[- ]?end|javascript|react|angular|\bvue\b|button|dialog|layout|render|page|widget|theme|icon|swing|gui"),
]


def classify_task_type(df: pd.DataFrame) -> pd.Series:
    """Keyword-label each issue with one of the app's 6 categories using issue type, components, labels and
    title. Unmatched issues become 'Backend' (the app's default issue_type). This is a documented proxy: Jira
    has no field that says Backend/Frontend/QA/..."""
    blob = _text_blob(df)
    out = pd.Series("Backend", index=df.index, dtype=object)
    assigned = pd.Series(False, index=df.index)
    for label, pat in _TYPE_PATTERNS:
        m = blob.str.contains(pat, regex=True, na=False) & ~assigned
        out[m] = label
        assigned |= m
    return out


def clean_completed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created"] = _to_datetime(df["created"])
    df["resolved"] = _to_datetime(df["resolved"])
    n0 = len(df)
    df = df[df["created"].notna() & df["resolved"].notna() & (df["resolved"] >= df["created"])]
    log(f"Completed issues with valid created/resolved dates: {len(df):,} of {n0:,}")
    df["actual_days"] = (df["resolved"] - df["created"]).dt.total_seconds() / 86400.0
    return df


# =====================================================================================================
# 3. Duration forecaster (same artifacts as forecasting_engine.train, but fit on Itemlet)
# =====================================================================================================
def _type_index(t: str) -> int:
    return TASK_TYPES.index(t) if t in TASK_TYPES else len(TASK_TYPES)


def train_duration(df: pd.DataFrame, ngb_samples: int, ngb_estimators: int, seed: int = 42) -> dict:
    d = clean_completed(df)
    sp = pd.to_numeric(d["story_points"], errors="coerce")
    d = d[sp.notna() & (sp > 0) & (sp <= 100)].copy()
    d["story_points"] = np.maximum(1, np.round(sp[d.index])).astype(int)
    log(f"Issues with usable story points: {len(d):,}")
    d["task_type"] = classify_task_type(d)
    cap = float(min(np.percentile(d["actual_days"], 99), 120.0)) if len(d) else 0
    d = d[(d["actual_days"] <= cap)]
    d["actual_days"] = d["actual_days"].clip(lower=0.25)      # same floor the app uses
    n = len(d)
    log(f"Duration training set: {n:,} issues (dropped the slowest 1% / >{cap:.0f} days as parked tickets)")
    log("Task-type mix: " + ", ".join(f"{k}={v:,}" for k, v in d["task_type"].value_counts().items()))
    if n < DUR_MIN_SAMPLES:
        raise SystemExit(f"Only {n} usable rows for the duration model (need {DUR_MIN_SAMPLES}).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    # --- empirical baseline (same structure as forecasting_engine._build_baseline) ---
    ratios = (d["actual_days"] / d["story_points"].clip(lower=1))
    all_ratios = ratios.values
    stats = {}
    for t in TASK_TYPES:
        arr = ratios[d["task_type"] == t].values
        if len(arr) < 5:
            arr = all_ratios
        stats[t] = {"p50_ratio": float(np.percentile(arr, 50)), "p80_ratio": float(np.percentile(arr, 80)),
                    "p95_ratio": float(np.percentile(arr, 95)), "mean_ratio": float(arr.mean()),
                    "std_ratio": float(arr.std()) if len(arr) > 1 else float(arr.mean()) * 0.3,
                    "n": int((d["task_type"] == t).sum())}
    baseline = {"status": "baseline_empirical", "model_version": "duration-baseline-v1.0", "n_samples": n,
                "min_required": DUR_MIN_SAMPLES, "by_type": stats, "data_source": "itemlet"}
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ImportError:
        meta = {"status": "baseline_empirical", "model_version": "duration-ngboost-v1.0", "n_samples": n,
                "data_source": "itemlet",
                "note": "ngboost not installed - wrote the empirical baseline fitted on Itemlet instead. "
                        "pip install -r requirements-itemlet.txt and re-run to train NGBoost."}
        json.dump(meta, open(NGB_META_PATH, "w"), indent=2)
        log("ngboost not installed -> wrote Itemlet empirical baseline only.")
        return meta

    if len(d) > ngb_samples:
        d = d.sample(n=ngb_samples, random_state=seed)
    # feature layout MUST match forecasting_engine: [story_points, task_type_index, estimated_days]
    X = np.column_stack([d["story_points"].values.astype(float),
                         d["task_type"].map(_type_index).values.astype(float),
                         d["story_points"].values / 2.5])        # estimated_days default the app uses
    y = d["actual_days"].values.astype(float)
    if "project" in d.columns and d["project"].nunique() >= 10:
        # hold out whole PROJECTS so validation measures generalisation to unseen teams
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, d["project"]))
        split_kind = "held-out projects"
    else:
        tr, va = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
        split_kind = "random rows"
    log(f"Training NGBoost on {len(tr):,} rows ({ngb_estimators} trees), validating on {len(va):,} ({split_kind}) ...")
    t0 = time.time()
    model = NGBRegressor(Dist=Normal, n_estimators=ngb_estimators, verbose=False, random_state=seed)
    model.fit(X[tr], y[tr])
    val_pred = model.predict(X[va])
    dist = model.pred_dist(X[va])
    mae = float(mean_absolute_error(y[va], val_pred))
    rmse = float(mean_squared_error(y[va], val_pred) ** 0.5)
    nll = float(-np.mean(dist.logpdf(y[va])))
    cov80 = float(np.mean((y[va] >= dist.dist.ppf(0.10)) & (y[va] <= dist.dist.ppf(0.90))))
    training_median = float(np.median(y[tr]))
    naive_mae = float(np.mean(np.abs(y[va] - training_median)))
    import pickle
    with open(NGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    from app.ai import baseline_gate
    clusters = d["project"].values[va] if split_kind == "held-out projects" else np.arange(len(va))
    gate = baseline_gate.evaluate_gate(
        component="duration", model_name="NGBoost", baseline_name="training_median",
        y_true=y[va], pred_model=val_pred, pred_base=np.full(len(va), training_median),
        clusters=clusters, dataset=f"D2-itemlet ({split_kind})",
    )
    baseline_gate.save_gate(gate)
    log(f"Baseline gate (duration): {gate['status']}  delta MAE {gate.get('delta')} "
        f"[{gate.get('ci_low')}, {gate.get('ci_high')}] over {gate.get('n_clusters')} clusters")
    segment_gate = baseline_gate.evaluate_gate_by_segment(
        "duration", "NGBoost", "training_median", y[va], val_pred, np.full(len(va), training_median), clusters,
        [baseline_gate.duration_segment(sp) for sp in X[va, 0]], f"D2-itemlet ({split_kind})", global_record=gate)
    baseline_gate.save_segment_gate(segment_gate)
    log("Segment gate (duration): " + ", ".join(f"{k}={v['status']}" for k, v in segment_gate["segments"].items()))
    meta = {
        "status": "trained", "gate": gate, "segment_gate": segment_gate,
        "training_median_days": round(training_median, 4),
        "model_version": "duration-ngboost-v1.0", "data_source": "itemlet",
        "n_samples": n, "n_train": int(len(tr)), "n_val": int(len(va)), "validation_split": split_kind,
        "min_required": DUR_MIN_SAMPLES,
        "mae_days": round(mae, 3), "rmse_days": round(rmse, 3), "negative_log_likelihood": round(nll, 3),
        "interval_80_coverage": round(cov80, 3), "naive_median_mae_days": round(naive_mae, 3),
        "train_seconds": round(time.time() - t0, 1),
        "note": f"Trained on {len(tr):,} real Jira issues from the Itemlet dataset, validated on {len(va):,} "
                f"({split_kind}). Target = days from created to resolved. naive_median_mae_days is the error "
                "of always predicting the training median, so you can see what the model adds.",
    }
    json.dump(meta, open(NGB_META_PATH, "w"), indent=2)
    log(f"Duration model: MAE {mae:.2f}d (naive {naive_mae:.2f}d), 80% interval coverage {cov80:.1%}")
    return meta


# =====================================================================================================
# 4. BiGRU workload model: real per-developer daily activity sequences from Jira history
# =====================================================================================================
_DAY = np.timedelta64(1, "D")


def _days(ts: pd.Series) -> np.ndarray:
    return ts.values.astype("datetime64[D]").astype("int64")


def developer_series(created: pd.Series, resolved: pd.Series, points: np.ndarray, max_days: int,
                     active_cap_days: int) -> Optional[dict]:
    """One developer's daily feature matrix (T,10), next-day targets source arrays, built the same way the
    app's DailyWorkloadSnapshot feeds the BiGRU (see workload_engine.build_sequences_for_training)."""
    c_day, r_day = _days(created), _days(resolved)
    start = np.maximum(c_day, r_day - (active_cap_days - 1))   # a ticket counts as "active" for at most N days
    day1 = int(r_day.max())
    day0 = max(int(start.min()), day1 - max_days + 1)
    T = day1 - day0 + 1
    if T < bigru.SEQ_LEN + 2:
        return None
    keep = r_day >= day0
    start = np.maximum(start[keep], day0) - day0
    end = r_day[keep] - day0
    pts = points[keep]
    c_ts, r_ts = created[keep], resolved[keep]

    def interval_sum(s, e, w):                       # sum of w over intervals [s,e] for each day (len T)
        diff = np.zeros(T + 8)
        np.add.at(diff, s, w)
        np.add.at(diff, np.minimum(e + 1, T + 7), -w)
        return np.cumsum(diff)[:T]

    ones = np.ones(len(pts))
    active_pts = interval_sum(start, end, pts)
    active_cnt = interval_sum(start, end, ones)
    review_q = interval_sum(np.minimum(start + 14, end + 1), end, ones) if len(pts) else np.zeros(T)
    big = pts >= 8
    # "touched in trailing 7 days" == active on some day within the window == interval [start, end+6]
    high_cx = interval_sum(start[big], end[big] + 6, np.ones(big.sum())) if big.any() else np.zeros(T)
    ctx_sw = interval_sum(start, end + 6, ones)
    completed = np.bincount(end, weights=pts, minlength=T)[:T]

    # after-hours: share of this developer's logged events (ticket created / resolved) outside 09-18h or on weekends (UTC)
    ev_ts = pd.concat([c_ts, r_ts])
    ev_day = _days(ev_ts) - day0
    ok = (ev_day >= 0) & (ev_day < T)
    ev_off = ((ev_ts.dt.hour.values < 9) | (ev_ts.dt.hour.values >= 18) | (ev_ts.dt.weekday.values >= 5)).astype(float)
    tot_day = np.bincount(ev_day[ok], minlength=T)[:T].astype(float)
    off_day = np.bincount(ev_day[ok], weights=ev_off[ok], minlength=T)[:T]
    k = np.ones(7)
    tot7 = np.convolve(tot_day, k, "full")[:T]
    off7 = np.convolve(off_day, k, "full")[:T]
    after = np.where(tot7 > 0, off7 / np.maximum(tot7, 1), 0.0)

    weekday = (np.arange(T) + day0 + 3) % 7                      # epoch day 0 was a Thursday
    rest = ((weekday >= 5) & (tot_day == 0)).astype(float)
    consec = np.zeros(T)
    run = 0
    for i in range(T):
        run = 0 if rest[i] else run + 1
        consec[i] = run
    load_ratio = np.minimum(2.0, active_pts / SPRINT_CAPACITY)

    risk = np.clip(W_LOAD * np.minimum(1.0, load_ratio) + W_HIGH * np.minimum(1.0, high_cx / 3.0)
                   + W_AFTER * after + W_CTX * np.minimum(1.0, ctx_sw / 5.0)
                   - W_REST * (consec == 0).astype(float), 0.0, 1.0)
    feats = np.column_stack([completed, active_pts, active_cnt, review_q, high_cx, after, load_ratio,
                             np.zeros(T), rest, load_ratio])     # identical column order to the app
    return {"features": feats, "risk": risk, "completed": completed, "active_cnt": active_cnt}


def build_developer_sequences(df: pd.DataFrame, min_issues: int, max_devs: int, max_days: int,
                              active_cap_days: int) -> List[dict]:
    d = clean_completed(df)
    d = d[d["assignee"].notna()].copy()
    if "story_points" in d.columns:
        sp = pd.to_numeric(d["story_points"], errors="coerce")
        n_missing = int(sp.isna().sum())
        d["pts"] = sp.fillna(3).clip(lower=1, upper=100)   # app's default story_points is 3
        log(f"Story points missing on {n_missing:,} issues ({n_missing/max(1,len(d)):.0%}); using the app default of 3 there.")
    else:
        d["pts"] = 3.0
        log("No story-point column found: every issue counted as 3 points (the app default).")
    counts = d["assignee"].value_counts()
    devs = counts[counts >= min_issues].index[:max_devs]
    log(f"Developers with >= {min_issues} resolved issues: {len(counts[counts >= min_issues]):,} (using {len(devs):,})")
    seqs = []
    for dev, g in d[d["assignee"].isin(set(devs))].groupby("assignee", sort=False):
        s = developer_series(g["created"], g["resolved"], g["pts"].values.astype(float), max_days, active_cap_days)
        if s is not None:
            s["developer"] = str(dev)
            seqs.append(s)
    log(f"Built daily activity sequences for {len(seqs):,} developers "
        f"({sum(len(s['risk']) for s in seqs):,} developer-days).")
    return seqs


def make_windows(seqs: List[dict], stride: int, return_groups: bool = False):
    L = bigru.SEQ_LEN
    X, yr, yv, cur, grp = [], [], [], [], []
    for gi, s in enumerate(seqs):
        f = s["features"]
        T = len(f)
        w = np.lib.stride_tricks.sliding_window_view(f, L, axis=0).transpose(0, 2, 1)   # (T-L+1, L, 10)
        idx = np.arange(0, T - L)                           # window ending at t=idx+L-1 has a next day
        idx = idx[::stride]
        t_end = idx + L - 1
        active = w[idx][:, :, 2].sum(axis=1) > 0
        idx, t_end = idx[active], t_end[active]
        if len(idx) == 0:
            continue
        X.append(w[idx])
        yr.append(s["risk"][t_end + 1])
        yv.append(s["completed"][t_end + 1])
        cur.append(s["risk"][t_end])
        grp.append(np.full(len(idx), s.get("developer", gi), dtype=object))
    if not X:
        empty = (np.zeros((0, L, bigru.INPUT_SIZE)),) + (np.zeros(0),) * 3
        return empty + (np.zeros(0, dtype=object),) if return_groups else empty
    out = (np.concatenate(X), np.concatenate(yr), np.concatenate(yv), np.concatenate(cur))
    return out + (np.concatenate(grp),) if return_groups else out


def train_bigru(df: pd.DataFrame, min_issues: int, max_devs: int, max_days: int, active_cap_days: int,
                stride: int, max_windows: int, epochs: int, batch: int, lr: float, patience: int,
                seed: int = 7) -> dict:
    import copy
    seqs = build_developer_sequences(df, min_issues, max_devs, max_days, active_cap_days)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seqs))
    n_val_dev = max(1, int(0.2 * len(seqs)))
    val_seqs = [seqs[i] for i in order[:n_val_dev]]
    tr_seqs = [seqs[i] for i in order[n_val_dev:]]       # split by DEVELOPER: no person appears in both sets
    Xtr, ytr_r, ytr_v, ctr = make_windows(tr_seqs, stride)
    Xva, yva_r, yva_v, cva, gva = make_windows(val_seqs, stride, return_groups=True)
    if len(Xtr) > max_windows:
        pick = rng.choice(len(Xtr), max_windows, replace=False)
        Xtr, ytr_r, ytr_v, ctr = Xtr[pick], ytr_r[pick], ytr_v[pick], ctr[pick]
    if len(Xva) > max_windows // 4:
        pick = rng.choice(len(Xva), max_windows // 4, replace=False)
        Xva, yva_r, yva_v, cva, gva = Xva[pick], yva_r[pick], yva_v[pick], cva[pick], gva[pick]
    n = len(Xtr) + len(Xva)
    log(f"BiGRU windows: {len(Xtr):,} train ({len(tr_seqs)} developers) / {len(Xva):,} validation ({len(val_seqs)} developers)")
    if n < bigru.MIN_TRAINING_SAMPLES * 5 or len(Xva) == 0:
        raise SystemExit(f"Only {n} usable windows - not enough real data for the BiGRU.")

    vel_cap = float(np.percentile(ytr_v, 99)) or 1.0       # keep rare giant days from dominating the MSE
    ytr_v, yva_v = np.minimum(ytr_v, vel_cap), np.minimum(yva_v, vel_cap)
    mean = Xtr.reshape(-1, bigru.INPUT_SIZE).mean(axis=0)
    std = Xtr.reshape(-1, bigru.INPUT_SIZE).std(axis=0) + 1e-6
    Xtr_n, Xva_n = (Xtr - mean) / std, (Xva - mean) / std

    model = bigru.BiGRUWorkloadModel()
    best, best_val, bad, history = None, np.inf, 0, []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr_n))
        rl = vl = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            r, v = model.train_step(Xtr_n[b], ytr_r[b], ytr_v[b], lr)
            rl += r * len(b); vl += v * len(b)
        vr, vv = model.predict(Xva_n)
        val_r, val_v = float(np.mean((vr - yva_r) ** 2)), float(np.mean((vv - yva_v) ** 2))
        history.append({"epoch": ep, "train_risk_mse": round(rl / len(perm), 5), "train_vel_mse": round(vl / len(perm), 5),
                        "val_risk_mse": round(val_r, 5), "val_vel_mse": round(val_v, 5)})
        log(f"  epoch {ep+1:>2}/{epochs}  train risk {rl/len(perm):.4f}  val risk {val_r:.4f}  val vel {val_v:.3f}")
        score = val_r + 0.05 * val_v
        if score < best_val - 1e-5:
            best, best_val, bad = copy.deepcopy(model), score, 0
        else:
            bad += 1
            if bad >= patience:
                log("  early stopping (validation stopped improving)")
                break
    model = best or model
    vr, vv = model.predict(Xva_n)
    final_r, final_v = float(np.mean((vr - yva_r) ** 2)), float(np.mean((vv - yva_v) ** 2))
    # honest reference points so the number can be judged
    mean_risk_mse = float(np.mean((ytr_r.mean() - yva_r) ** 2))
    persist_mse = float(np.mean((cva - yva_r) ** 2))
    mean_vel_mse = float(np.mean((ytr_v.mean() - yva_v) ** 2))
    model.save()
    from app.ai import baseline_gate
    gate = baseline_gate.evaluate_gate(
        component="risk", model_name="BiGRU", baseline_name="persistence",
        y_true=yva_r, pred_model=vr, pred_base=cva, clusters=gva,
        dataset="D3-itemlet (held-out grouping units)",
    )
    baseline_gate.save_gate(gate)
    log(f"Baseline gate (risk): {gate['status']}  delta MAE {gate.get('delta')} "
        f"[{gate.get('ci_low')}, {gate.get('ci_high')}] over {gate.get('n_clusters')} units")
    segment_gate = baseline_gate.evaluate_gate_by_segment(
        "risk", "BiGRU", "persistence", yva_r, vr, cva, gva, [baseline_gate.risk_segment(r) for r in cva],
        "D3-itemlet (held-out grouping units)", global_record=gate)
    baseline_gate.save_segment_gate(segment_gate)
    log("Segment gate (risk): " + ", ".join(f"{k}={v['status']}" for k, v in segment_gate["segments"].items()))
    meta = {
        "status": "trained", "gate": gate, "segment_gate": segment_gate, "model_version": "allocation-bigru-v1.0", "data_source": "itemlet",
        "n_samples": int(n), "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "min_required": bigru.MIN_TRAINING_SAMPLES,
        "n_developers_train": len(tr_seqs), "n_developers_val": len(val_seqs),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "final_train_risk_mse": history[-1]["train_risk_mse"], "final_val_risk_mse": round(final_r, 5),
        "final_train_vel_mse": history[-1]["train_vel_mse"], "final_val_vel_mse": round(final_v, 5),
        "baseline_val_risk_mse_predict_mean": round(mean_risk_mse, 5),
        "baseline_val_risk_mse_persistence": round(persist_mse, 5),
        "baseline_val_vel_mse_predict_mean": round(mean_vel_mse, 5),
        "history": history,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "train_seconds": round(time.time() - t0, 1),
        "note": f"Trained on {len(Xtr):,} real 14-day developer windows from the Itemlet Jira history and validated "
                f"on {len(Xva):,} windows from {len(val_seqs)} developers never seen in training. Targets are the "
                "next day's baseline recovery risk and completed story points (the app's own self-supervised "
                "targets). Compare final_val_risk_mse with baseline_val_risk_mse_persistence / _predict_mean.",
    }
    json.dump(meta, open(bigru.META_PATH, "w"), indent=2)
    log(f"BiGRU: val risk MSE {final_r:.4f} (predict-mean {mean_risk_mse:.4f}, persistence {persist_mse:.4f}); "
        f"val velocity MSE {final_v:.3f} (predict-mean {mean_vel_mse:.3f})")
    return meta


# =====================================================================================================
# 5. CLI
# =====================================================================================================
def backup_model_store():
    files = ["bigru_weights.npz", "bigru_meta.json", "ngboost_duration.pkl", "ngboost_duration_meta.json",
             "duration_baseline.json"]
    existing = [f for f in files if os.path.exists(os.path.join(MODEL_DIR, f))]
    if not existing:
        return
    dst = os.path.join(MODEL_DIR, "_backup_before_itemlet_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst)
    for f in existing:
        shutil.copy2(os.path.join(MODEL_DIR, f), dst)
    log(f"Backed up previous (seed-trained) models to {os.path.relpath(dst, HERE)}")


def verify_artifacts():
    log("Verifying the app can load what was just trained ...")
    st = bigru.get_status()
    print(f"    bigru status: {st.get('status')} ({st.get('n_samples')} windows)")
    if st.get("status") == "trained":
        w = np.zeros((bigru.SEQ_LEN, bigru.INPUT_SIZE)); w[:, 1] = 20; w[:, 2] = 3; w[:, 6] = 0.5; w[:, 9] = 0.5
        print(f"    bigru predict_for_window(moderate-load week) -> {bigru.predict_for_window(w)}")
    try:
        from app.ai import forecasting_engine as fe
        print(f"    duration status: {fe.get_status().get('status')}")
        for sp, tp in [(3, "Backend"), (8, "Frontend")]:
            print(f"    forecast {sp}sp {tp}: {fe.predict(sp, tp)}")
    except Exception as e:
        print(f"    (duration predict check skipped: {type(e).__name__}: {e})")


def main(argv=None):
    import argparse
    default_dir = os.path.join(os.path.dirname(HERE), "data", "itemlet")
    ap = argparse.ArgumentParser(description="Train AdaptiveFlow's models on the Itemlet dataset.")
    ap.add_argument("--data", default=os.path.join(default_dir, "itemlet_dataset.parquet"))
    ap.add_argument("--dictionary", default=os.path.join(default_dir, "data_dictionary.csv"))
    ap.add_argument("--only", choices=["all", "duration", "bigru"], default="all")
    ap.add_argument("--inspect", action="store_true", help="print the column mapping and exit")
    ap.add_argument("--map", action="append", default=[], metavar="FIELD=COLUMN",
                    help="override a detected column, e.g. --map story_points=Story_Points. Fields: "
                         + ", ".join(LOGICAL_FIELDS))
    ap.add_argument("--max-rows", type=int, default=None, help="random subsample of the dataset (quick test runs)")
    ap.add_argument("--no-backup", action="store_true")
    g = ap.add_argument_group("duration model")
    g.add_argument("--ngb-samples", type=int, default=60000)
    g.add_argument("--ngb-estimators", type=int, default=200)
    b = ap.add_argument_group("BiGRU")
    b.add_argument("--min-issues", type=int, default=40, help="min resolved issues for a developer to be used")
    b.add_argument("--max-devs", type=int, default=3000)
    b.add_argument("--max-days", type=int, default=540, help="most recent days per developer")
    b.add_argument("--active-cap-days", type=int, default=30,
                   help="a ticket counts toward load for at most this many days before its resolution")
    b.add_argument("--stride", type=int, default=3)
    b.add_argument("--max-windows", type=int, default=120000)
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--lr", type=float, default=0.05)
    b.add_argument("--patience", type=int, default=5)
    a = ap.parse_args(argv)

    overrides = dict(m.split("=", 1) for m in a.map)
    need = {"all": sorted(set(REQUIRED_DURATION + REQUIRED_BIGRU)), "duration": REQUIRED_DURATION,
            "bigru": REQUIRED_BIGRU}[a.only]
    if a.inspect:
        path = a.data
        cols = (__import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(path).schema.names
                if path.endswith(".parquet") else list(pd.read_csv(path, nrows=0).columns))
        print_column_report(cols, resolve_columns(cols, overrides), a.dictionary)
        print("\nAll columns:\n  " + "\n  ".join(cols))
        return
    # the BiGRU needs 'assignee'; if the dataset lacks one, still train the duration model
    try:
        df, mapping = load_dataset(a.data, overrides, a.dictionary, a.max_rows, need)
    except SystemExit as e:
        if a.only == "all" and "assignee" in str(e):
            log("No assignee column -> training the duration model only (use --map assignee=<col> to fix).")
            a.only = "duration"
            df, mapping = load_dataset(a.data, overrides, a.dictionary, a.max_rows, REQUIRED_DURATION)
        else:
            raise
    if not a.no_backup:
        backup_model_store()
    results = {}
    if a.only in ("all", "duration"):
        results["duration"] = train_duration(df, a.ngb_samples, a.ngb_estimators)
    if a.only in ("all", "bigru"):
        results["bigru"] = train_bigru(df, a.min_issues, a.max_devs, a.max_days, a.active_cap_days, a.stride,
                                       a.max_windows, a.epochs, a.batch, a.lr, a.patience)
    verify_artifacts()
    log("Done. Restart the backend (uvicorn) and open Team / Simulator - they now use the Itemlet-trained models.")


if __name__ == "__main__":
    main()
