from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import models
from .database import engine, SessionLocal, run_light_migrations
from . import seed as seed_module
from .ai import rules_engine
from .routers import (
    teams, ideas, issues, absence, recommendations, dashboard, attendance, auth,
    epics, labels_components, reallocation, simulator, audit,
)

models.Base.metadata.create_all(bind=engine)
run_light_migrations(engine, models.Base)  # adds any new columns to already-existing tables safely

app = FastAPI(title="AdaptiveFlow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams.router)
app.include_router(ideas.router)
app.include_router(issues.router)
app.include_router(absence.router)
app.include_router(recommendations.router)
app.include_router(dashboard.router)
app.include_router(attendance.router)
app.include_router(auth.router)
app.include_router(epics.router)
app.include_router(labels_components.router)
app.include_router(reallocation.router)
app.include_router(simulator.router)
app.include_router(audit.router)


@app.on_event("startup")
def run_seed():
    db = SessionLocal()
    try:
        seed_module.seed(db)
        rules_engine.ensure_default_rules(db)  # safe no-op if already seeded
    finally:
        db.close()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AdaptiveFlow API"}
