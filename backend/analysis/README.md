# Analysis scripts used in the paper

Run from `backend/` after `evaluate_itemlet.py` has written `results/`:

| Script | Paper | Command |
|---|---|---|
| `null_simulation.py` | Table "False promotions in 1,000 simulated data sets" and the cross-fitted rule comparison | `python analysis/null_simulation.py . results results/null_simulation.json` |
| `gate_variants.py` | Clustered vs row bootstrap vs threshold on the real out-of-fold files | `python analysis/gate_variants.py . results results/gate_variants.json` |
| `receipt_overhead.py` | Receipt size, replay and verification time (D1) | `python analysis/receipt_overhead.py` |
| `replay_stress.py` | Replay through the live code path on 30 randomized organizations, with re-signed manipulations | `python analysis/replay_stress.py --runs 30` |

`evaluate_extended.py` (in `backend/`) adds richer leakage-safe duration features, a chronological split, and the
next-day velocity task (an observed outcome). It accepts any Jira export with the same columns, e.g. TAWOS:
`python evaluate_extended.py --data tawos_issues.csv --tag tawos --only duration`.
