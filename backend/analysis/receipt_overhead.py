"""Receipt overhead on the synthetic organization (D1): size per receipt, replay and verification time.
Run from backend/:  python analysis/receipt_overhead.py"""
import json, os, statistics as st, sys, tempfile, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tests")
import conftest  # noqa: F401  (temporary database and model store)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app import models, seed as seed_module, schemas
from app.ai import rules_engine, bigru, workload_engine, audit_ledger, decision_receipt as dr, assignment_engine as ae
from app.routers import absence

eng = create_engine(f"sqlite:///{tempfile.mkdtemp()}/o.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(eng); db = sessionmaker(bind=eng)()
seed_module.seed(db); rules_engine.ensure_default_rules(db)
bigru.train(workload_engine.build_sequences_for_training(db))
issues = db.query(models.Issue).filter(models.Issue.status != "Done").all()
t = time.perf_counter(); recs = [ae.recommend_for_issue(db, i) for i in issues]; t_rec = (time.perf_counter() - t) / len(issues)
sizes = [len(audit_ledger.canonical_json(r["receipt"]).encode()) for r in recs]
ncand = [len(r["receipt"]["inputs"]) for r in recs]
rt = []
for r in recs:
    t = time.perf_counter(); assert dr.replay(r["receipt"])["status"] == "REPLAY_MATCH"; rt.append((time.perf_counter() - t) * 1000)
emps = db.query(models.Employee).all()
load = {e.id: db.query(models.Issue).filter(models.Issue.assignee_id == e.id, models.Issue.status != "Done").count() for e in emps}
absence.mark_absent(schemas.AbsenceRequest(employee_id=max(load, key=load.get)), db); db.expire_all()
vt = []
for e in db.query(models.AuditEvent).filter(models.AuditEvent.event_type == "REALLOCATION_RECOMMENDED"):
    t = time.perf_counter(); assert dr.verify_decision(db, e.event_id)["status"] == "VERIFIED"; vt.append((time.perf_counter() - t) * 1000)
print(json.dumps({"decisions": len(recs), "candidates_per_receipt": [min(ncand), max(ncand)],
                  "receipt_kb_mean": round(st.mean(sizes) / 1024, 1),
                  "bytes_per_candidate": round(st.mean(s / n for s, n in zip(sizes, ncand))),
                  "replay_ms_median": round(st.median(rt), 2), "verify_ms_median": round(st.median(vt), 1),
                  "recommend_ms_per_issue": round(t_rec * 1000, 1)}, indent=1))
