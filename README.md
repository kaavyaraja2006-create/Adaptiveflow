# AdaptiveFlow

A real, working AI-native Jira-style project intelligence platform: FastAPI + SQLAlchemy + SQLite backend, React + TypeScript + Tailwind frontend. Every AI feature below is a genuine, explainable calculation, trained model, or simulation over real seeded data — no hardcoded numbers pretending to be AI, no fabricated accuracy metrics, and no live LLM call required to run it.

This is a prototype built to demonstrate one connected pipeline, not three separate bolted-on features:

```
Live engineering data → data-quality gate → risk & sustainable-workload analysis
    → digital-twin simulation → human-aware allocation → explainable recommendation
    → manager approval → real project action → cryptographically verifiable audit record
```

---

## What's real in this build

### Core PM platform
- **Epics, Board, Backlog, Issue Detail** — real hierarchy/progress rollup, drag-and-drop persisted to the DB, comments/watchers/links/labels/components, a full activity log auto-written on every field change.
- **Idea AI Analysis** — structured extraction, impact/effort/priority prediction, duplicate detection.
- **Team & capability profiles** — computed live from real work history.
- **Check-In (biometric + manual)** — face enrollment/matching runs entirely client-side (face-api.js); no photo or video ever reaches the backend, only a 128-number descriptor.
- **Demo login** — pick a persona (Admin/PM/Developer/Viewer). Explicitly **not real security**.

### Feature 1 — Human-Aware Allocation Engine
- **Data Quality Gate** (`ai/data_quality.py`) — flags stale tickets, invalid story points, duplicate active assignments, missing skills before any of this data is trusted by an AI pipeline.
- **Sustainable Workload / Recovery Risk** (`ai/workload_engine.py`) — *never* called a burnout/mental-health detector. Computed from transparent work-pattern signals only (sprint load ratio, task-switching, after-hours activity, rest days) — never webcam emotion analysis, keystroke tracking, or health data. Employees can opt out (`workload_opt_out`); opted-out employees are excluded from scoring everywhere, with no exceptions.
- **BiGRU sequence model** (`ai/bigru.py`) — a genuine bidirectional GRU with full backpropagation-through-time, implemented from scratch in NumPy (see "Why NumPy instead of PyTorch" below). Trains on real 14-day activity windows derived from seeded history. **Honesty contract + baseline gate** (`ai/baseline_gate.py`): below 40 windows it reports `insufficient_data`. A trained model's risk is only *served* if it beats persistence on held-out data (paired MAE difference, 95% CI clustered by employee/unit, upper bound < 0) — otherwise the status is `trained_bigru_gated_demoted`, the deterministic value is served, and both values plus the gate record are returned. The NGBoost duration forecaster is gated the same way against the training median; when demoted, the median is served as the point forecast and the NGBoost interval is kept, labelled. **Segment-wise gate:** the same test is also run inside pre-declared segments (story-point band for duration; today's risk band for workload risk), with Holm correction across segments; a segment with too little held-out evidence inherits the global verdict. Each prediction is served by its own segment's verdict, so a model can be trusted for large tickets but not small ones.
- **Symbolic rule engine, R1–R10** (`ai/rules_engine.py`) — transparent, manager-editable (`GET/PATCH /api/allocation-rules`), stored in the DB, sitting *above* the scoring layer: a rule-excluded candidate can never win purely on score.
- **AllocationScore v2** (`ai/assignment_engine.py`) — `clip01(0.30·skill + 0.20·history + 0.15·availability + 0.15·confidence + 0.10·fairness + 0.10·urgency − 0.30·(0.55·load + 0.45·recoveryRisk) − rulePenalty/100 − 0.15·transferCost)`. The workload term can remove at most 30 points (v1: 100), and fairness is counted once (v1 also subtracted `min(25, 8·reassignments)`). Every candidate carries `score_version` and a per-term `score_breakdown` in points; plus a knowledge-transfer-cost penalty (domain familiarity × code ownership × task complexity heuristic — explicitly labeled as a heuristic since this build has no Git integration to derive real code ownership from).
- Every recommendation ships **both** a model explanation (which quantitative factors helped/hurt) and a policy explanation (which rules fired and why).

### Feature 4 — Project Digital Twin
- **Dependency graph** (`ai/graph_engine.py`, NetworkX) — built from real `IssueLink` data, employee skills, sprint membership. Honestly scoped: no Repository/PR/Build-Pipeline nodes are fabricated since this build has no Git/CI integration.
- **Uncertainty-aware duration forecasting** (`ai/forecasting_engine.py`) — NGBoost trained on real historical `WorkHistoryItem` completions, producing a full predictive distribution (p50/p80/p95), not a point estimate. **Optional dependency** (`requirements-ml.txt`) — without it, an honest empirical-percentile baseline is used instead, clearly labeled `baseline_empirical`.
- **Monte Carlo simulator** (`ai/monte_carlo_engine.py`) — samples task durations per iteration, respects real dependencies and per-developer capacity, and produces sprint-miss probability + p50/p80/p95 completion from the actual distribution of simulated outcomes. Supports all 7 what-if scenario types from the spec: developer absence, add team member, scope change, priority change, dependency failure, capacity reduction, process change.
- **SprintRisk composite** (`ai/digital_twin.py`) — `0.35·P(miss) + 0.20·criticalDependencyRisk + 0.15·workloadRisk + 0.15·blockedTaskRisk + 0.15·uncertainty`, every term computed from the real simulation/graph/workload output above.
- **Connected to the allocation engine**: `GET /api/ai/reallocation/{task_id}/simulate-candidates` re-simulates the sprint with each top candidate actually assigned and reports each one's simulated sprint-success probability — not just their allocation score restated.

### Feature 2 — Immutable AI Decision Ledger
- **SHA-256 hash chain + Ed25519 signatures** (`ai/audit_ledger.py`) — every consequential AI decision (`REALLOCATION_RECOMMENDED`, `RECOMMENDATION_APPROVED/MODIFIED/REJECTED`, `TASK_REASSIGNED`, `SIMULATION_STARTED/COMPLETED`) is recorded as a canonical-JSON-hashed, chained, signed, append-only event. Verified end-to-end including a deliberate tamper test (see below).
- **Manager approval is mandatory** — nothing reassigns automatically outside the pre-baked Check-In demo flow; `POST /api/absence/reassign/{issue}/{employee}?decision=approved|modified` and `POST /api/absence/reject/{issue}` are both explicit, both logged.
- **Replayable decision receipts** (`ai/decision_receipt.py`) — every `REALLOCATION_RECOMMENDED` event carries, inside its signed payload, a receipt with each candidate's raw inputs, the recovery risk that was served *and* the baseline/model values the gate chose between, the rule set in force (+hash), the baseline-gate records in force (+hash), the scoring constants and a fingerprint of the scoring code. `GET /api/audit/verify/decision/{event_id}` checks the ledger (hash, signature, chain, columns) **and** re-runs the decision offline from the receipt: it confirms that a demoted model could not have influenced the score, that every score and fired rule recomputes, and that the ranking reproduces the recommendation. Manager approvals/rejections and candidate simulations are linked to the recommendation by event hash and receipt hash; an approval that picks someone other than the recommended person is recorded as `RECOMMENDATION_MODIFIED` whatever the client declared. The timestamp is now hashed, and the queryable columns are checked against their hashed copies. Known limit: a holder of the signing key who fabricates inputs *and* re-runs the scorer consistently produces a receipt that replays — only external anchoring closes that (see `tests/test_decision_receipt.py`, attack A7).
- **Daily Merkle batching** (`build_daily_batch`) — batches unbatched events into one root hash instead of one anchor per event.
- **Hyperledger Fabric — optional, and honestly a stub** (`ai/fabric_stub.py`): this sandbox has no way to run a real Fabric peer/orderer network, and the spec is explicit the app "must remain runnable without Hyperledger Fabric." The stub implements the exact `registerAuditRoot/getAuditRoot/verifyAnchor` chaincode interface backed by local storage, so swapping in a real Fabric SDK client later only touches this one file.
- **Audit Verification page** (frontend `/audit`) — look up any task's full decision trail, verify a single event (recomputes the hash, checks the chain link, checks the signature) or the whole chain, and inspect Merkle batches.

---

## Why NumPy instead of PyTorch, and NGBoost as optional

The spec's suggested stack is PyTorch (BiGRU) and NGBoost (duration forecasting). Both were actually attempted:

- **PyTorch**: a CPU install pulls ~2GB of CUDA/toolkit dependencies. The first install attempt in this build's history ran the sandbox out of disk mid-install. Rather than fake a "trained" status or skip the sequence model, `ai/bigru.py` is a real bidirectional GRU with full backpropagation-through-time, implemented from scratch in NumPy (~17MB dependency, already needed) — verified to have correct gradients (loss decreases monotonically across 120 epochs on held-out validation data). If PyTorch becomes viable in a given deployment, only this file would need to change.
- **NGBoost**: much lighter (~50KB core package), but its dependency chain (scipy/pandas/scikit-learn/matplotlib, ~300-400MB) is still meaningful weight for a prototype. It's installed and working (see `requirements-ml.txt`) with real trained metrics (MAE 0.91 days, 80%-interval coverage 95.2% on held-out data at last training run) — but kept **optional**, with an honest empirical-baseline fallback (`ai/forecasting_engine.py`) if it isn't installed, following the same pattern as Hyperledger Fabric.

## Tests

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt
python -m pytest tests -q
```

`tests/test_rules.py` (16 rule tests, Table 5), `tests/test_allocation_score.py` (reproduces Tables 8-10 from the shipped code), `tests/test_baseline_gate.py` (gate logic and the demoted-forecast path), `tests/test_pipeline.py` (all active seeded issues; writes `results/pipeline_v2.json`), `tests/test_segment_gate.py` (Holm, segment detection, false-promotion control, per-request serving), `tests/test_decision_receipt.py` (replays every decision of a real absence-recovery run and runs ten attacks; writes `results/receipt_report.json`). Tests use a temporary database and model store.

`python evaluate_itemlet.py` now also writes per-row out-of-fold predictions (`results/*_oof.csv`), a training-median baseline row, the gate decisions (`results/gate_duration.csv`, `results/gate_risk.csv`), the segment-wise verdicts (`results/gate_*_segments.csv`) and a cross-fitted comparison of serving policies — always-baseline, always-model, global gate, segment gate (`results/gate_*_policy.csv`).

## What's intentionally out of scope

Roadmap/Gantt views, Goals/Initiatives with alignment scoring, Releases/versions, AI-driven sprint *planning*, full RBAC/notifications, global search, real Git/CI-CD integration (so Repository/PR/Build-Pipeline graph nodes and true code-ownership data don't exist), and a live Hyperledger Fabric network were all left out deliberately, to keep everything else fully real rather than mixing genuine and fake functionality.

---

## Running it locally

### Requirements
- **Python 3.11 or 3.12** (not 3.13/3.14 — some dependencies don't have prebuilt wheels yet)
- **Node.js 18+**

### 1. Backend

```bash
cd backend
python3 -m venv venv

# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1

pip install -r requirements.txt
# Optional — enables real NGBoost duration forecasting instead of the
# empirical baseline (adds ~300-400MB: scipy/pandas/scikit-learn):
pip install -r requirements-ml.txt

python -m uvicorn app.main:app --reload --port 8000
```

Leave this running. Check it worked: http://localhost:8000/api/health should return `{"status":"ok","service":"AdaptiveFlow API"}`. First startup seeds demo data, pre-generates 21 days of workload activity history, and trains the duration-forecasting model on real seeded `WorkHistoryItem` data — this can take 10-20 seconds the first time.

### 2. Frontend

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and pick any demo persona to log in.

### 3. Try the full connected demo flow

1. **Team** → notice Karthik Rao's elevated recovery-risk badge (heavy seeded load). Try the **opt-out toggle** and watch him disappear from the workload view and from allocation candidate pools everywhere.
2. **Team** → click **"Train BiGRU on seeded data"** and watch the model status switch from baseline to `trained_bigru` with real validation-loss numbers.
3. **Absence Recovery** → mark someone absent → review ranked candidates with model + policy explanations, rule exclusions, and recovery-risk badges → **Approve**.
4. **Simulator** → **Run Digital Twin** on a project to see live Monte Carlo sprint risk, p50/p80/p95 completion, and critical tasks.
5. **Audit Verification** → paste in the task ID you just reassigned → see the full `REALLOCATION_RECOMMENDED → RECOMMENDATION_APPROVED → TASK_REASSIGNED` chain → click through to verify a single event's hash/signature, or the whole chain at once.
6. **Decision History** → see the same decision from the manager's-eye view.

---

## Biometric check-in — an honest caveat

Face enrollment/matching run **entirely in the browser**. Only a 128-number descriptor reaches the backend — never a photo or video. Storing biometric data has real legal weight in many places (Illinois' BIPA, GDPR, etc.) — get your own legal/HR sign-off before pointing this at real employees. This is a working prototype, not a compliance-reviewed product.

## Swapping in a real database

```bash
export DATABASE_URL="postgresql://user:password@localhost/adaptiveflow"
```

No code changes needed. New columns added to existing tables since your last run are added automatically via a safe light-migration step in `database.py` (`ALTER TABLE ... ADD COLUMN`, never `DROP`) — your existing data is preserved.

## API reference (new in this build)

```
# Allocation (Phase 1-2)
POST   /api/ai/reallocation/analyze
GET    /api/ai/reallocation/{task_id}/candidates
GET    /api/ai/reallocation/{task_id}/simulate-candidates
GET    /api/employees/{id}/workload-profile
GET    /api/employees/{id}/recovery-risk
GET    /api/allocation-rules
PATCH  /api/allocation-rules/{code}
GET    /api/data-quality/scan
GET    /api/ai/models/status          # BiGRU status
POST   /api/ai/models/train           # BiGRU training

# Digital Twin (Phase 4-7)
POST   /api/simulator/run
POST   /api/simulator/whatif
GET    /api/simulator/graph/{project_id}
GET    /api/simulator/history/{project_id}
GET    /api/simulator/models/duration/status   # NGBoost/baseline status
POST   /api/simulator/models/duration/train

# Audit Ledger (Phase 8-13)
GET    /api/audit/events/{event_id}
GET    /api/audit/tasks/{task_id}
GET    /api/audit/projects/{project_id}
GET    /api/audit/verify/event/{event_id}
GET    /api/audit/verify/batch/{batch_id}
GET    /api/audit/verify/chain
POST   /api/audit/anchor/daily
GET    /api/audit/batches
GET    /api/audit/public-key

# Human-in-the-loop decisions
POST   /api/absence/reassign/{issue_id}/{employee_id}?decision=approved|modified
POST   /api/absence/reject/{issue_id}
```

## Project structure

```
backend/
  requirements.txt       # core app — lightweight, no ML extras
  requirements-ml.txt     # optional: ngboost, for real (vs. baseline) duration forecasting
  app/
    main.py               # FastAPI app + router wiring + light-migration on startup
    models.py              # SQLAlchemy schema
    schemas.py              # Pydantic contracts
    seed.py                  # demo data + auto-trains the duration model
    ai/
      capability_engine.py       # skill/success-rate profiles from history
      idea_engine.py              # duplicate detection, impact/effort/priority
      biometric_engine.py          # face descriptor matching
      risk_engine.py                # project health, delay, methodology, bottlenecks
      scenario_engine.py             # legacy lightweight what-if (still used by /api/simulate)
      data_quality.py                 # Phase 1: data-quality gate
      workload_engine.py               # Phase 1-3: sustainable workload / recovery risk
      rules_engine.py                   # Phase 2: symbolic rules R1-R10
      assignment_engine.py               # Phase 2: hybrid human-aware allocation
      bigru.py                            # Phase 3: from-scratch NumPy BiGRU
      graph_engine.py                      # Phase 4: NetworkX dependency graph
      forecasting_engine.py                 # Phase 4-5: NGBoost / baseline duration forecast
      monte_carlo_engine.py                  # Phase 6: Monte Carlo what-if simulation
      digital_twin.py                         # Phase 7: SprintRisk orchestrator + triggers
      audit_ledger.py                          # Phase 8-12: hash chain, Ed25519, Merkle
      fabric_stub.py                            # Phase 13: optional Fabric chaincode-shaped stub
      model_store/                               # trained weights, keys, metadata (gitignored)
    routers/               # one file per resource area, incl. reallocation.py, simulator.py, audit.py
frontend/
  src/
    api.ts                # typed API client
    pages/                 # ...AbsenceRecovery, Simulator, DecisionHistory, AuditVerification
    components/              # Sidebar, ConfidenceRing, Bits (incl. RecoveryRiskBadge)
```
