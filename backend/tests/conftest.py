"""
Test fixtures. Every test runs against a throw-away SQLite database and a
throw-away model_store directory, so running the suite never touches the
developer's adaptiveflow.db or trained models.
"""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

_TMP = tempfile.mkdtemp(prefix="adaptiveflow_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(_TMP, 'test.db')}")


def _redirect_model_store(path):
    from app.ai import bigru, forecasting_engine, baseline_gate
    os.makedirs(path, exist_ok=True)
    bigru.MODEL_DIR = path
    bigru.WEIGHTS_PATH = os.path.join(path, "bigru_weights.npz")
    bigru.META_PATH = os.path.join(path, "bigru_meta.json")
    forecasting_engine.MODEL_DIR = path
    forecasting_engine.NGB_MODEL_PATH = os.path.join(path, "ngboost_duration.pkl")
    forecasting_engine.NGB_META_PATH = os.path.join(path, "ngboost_duration_meta.json")
    forecasting_engine.BASELINE_PATH = os.path.join(path, "duration_baseline.json")
    baseline_gate.MODEL_DIR = path
    from app.ai import audit_ledger, fabric_stub
    audit_ledger.KEY_DIR = path
    audit_ledger.PRIVATE_KEY_PATH = os.path.join(path, "audit_signing_key.pem")
    audit_ledger.PUBLIC_KEY_META_PATH = os.path.join(path, "audit_public_key.json")
    fabric_stub.LEDGER_STUB_PATH = os.path.join(path, "fabric_stub_ledger.json")


_redirect_model_store(os.path.join(_TMP, "model_store"))


@pytest.fixture()
def db():
    """Empty schema with the default R1-R10 rules, no seeded organisation."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app import models  # noqa: F401  (registers tables)
    from app.ai import rules_engine
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    session = sessionmaker(bind=eng)()
    rules_engine.ensure_default_rules(session)
    yield session
    session.close()


@pytest.fixture(scope="session")
def seeded_db():
    """The application's own synthetic organisation (dataset D1), seeded once per test run."""
    from app.database import Base, SessionLocal, engine
    from app import models  # noqa: F401
    from app import seed as seed_module
    from app.ai import rules_engine
    Base.metadata.create_all(engine)
    session = SessionLocal()
    seed_module.seed(session)
    rules_engine.ensure_default_rules(session)
    yield session
    session.close()
