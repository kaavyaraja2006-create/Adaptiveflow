# Training AdaptiveFlow on the real Itemlet dataset

Nothing in the existing app was changed. Three new files were added: `backend/itemlet_training.py`,
`backend/train_itemlet.py`, `backend/requirements-itemlet.txt`.

## Steps
1. Download `itemlet_dataset.parquet` and `data_dictionary.csv` from https://zenodo.org/records/19411554
   into `data/itemlet/` (other Zenodo files are not needed).
2. ```
   cd backend
   pip install -r requirements.txt -r requirements-itemlet.txt
   python train_itemlet.py --inspect          # check the detected column mapping first
   python train_itemlet.py                     # train both models
   python -m uvicorn app.main:app --reload --port 8000
   ```
   If `--inspect` shows a wrong/missing column, fix it with e.g. `--map story_points=<column>`
   (names are in data_dictionary.csv). Quick trial run: `--max-rows 50000`. Needs Python 3.11/3.12.

## What gets trained (and what does not)
| Module | Trained on Itemlet? | How |
|---|---|---|
| Duration forecaster (NGBoost) | Yes | story points + task type -> days from created to resolved; validated on held-out projects. Feeds the Monte Carlo simulator. |
| BiGRU workload / recovery risk | Yes | 14-day per-developer activity windows rebuilt from assignee + created/resolved history; validated on developers never seen in training. |
| Monte Carlo, allocation score, rules, audit ledger | No parameters to learn | They consume the two models above. |
| Dependency graph | No | Itemlet has no issue-link table; the graph still uses your DB's IssueLink data. |

Previous (seed-trained) models are backed up in `backend/app/ai/model_store/_backup_before_itemlet_*`.
Results (MAE, 80% interval coverage, BiGRU validation MSE vs. simple baselines) are written to the `*_meta.json`
files and shown by the existing Team / Simulator status endpoints.

## Honest notes
- Task type (Backend/Frontend/QA/DevOps/Mobile/Security) is keyword-labelled from issue type, components, labels
  and title; unmatched issues are "Backend".
- Jira has no per-day workload data, so the BiGRU's daily features are derived proxies (e.g. after-hours = share of
  created/resolved timestamps outside 09-18 UTC / weekends; review queue = tickets open > 14 days). Missing story
  points count as 3 (the app default).
- The "Train" buttons in the UI retrain on the seeded demo DB and will overwrite these models; re-run
  `train_itemlet.py` afterwards.
