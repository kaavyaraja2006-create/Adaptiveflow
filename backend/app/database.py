"""
Database wiring. Defaults to a local SQLite file so the whole project runs
with zero external setup. Point DATABASE_URL at a real Postgres instance
(e.g. postgresql://user:pass@host/db) for production without changing any
other code — every model/query in this project goes through SQLAlchemy's
ORM layer, not raw SQL, so the swap is safe.
"""
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./adaptiveflow.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Simple "ALTER TABLE ... ADD COLUMN" style light-migration for columns added
# to models AFTER a database file has already been created by an earlier
# version of this app. `Base.metadata.create_all()` only creates missing
# TABLES, not missing COLUMNS on existing tables, so without this an
# existing local adaptiveflow.db would need to be deleted (losing demo
# data) every time a model gains a new field. This intentionally supports
# SQLite and Postgres only (both support plain ADD COLUMN) and only ever
# ADDS columns — it never drops or rewrites existing data.
_COLUMN_DEFAULTS_SQL = {
    "Boolean": "0",
    "Integer": "0",
    "Float": "0",
}


def run_light_migrations(target_engine, base):
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    with target_engine.begin() as conn:
        for table in base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand-new table — create_all() already handles this
            existing_cols = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_cols:
                    continue
                col_type = column.type.__class__.__name__
                sqlite_types = {"Integer": "INTEGER", "Float": "FLOAT", "Boolean": "BOOLEAN",
                                 "String": "VARCHAR", "Text": "TEXT", "DateTime": "DATETIME", "JSON": "JSON"}
                sql_type = sqlite_types.get(col_type, "VARCHAR")
                default_clause = ""
                if column.default is not None and getattr(column.default, "arg", None) is not None \
                        and not callable(column.default.arg):
                    default_val = column.default.arg
                    if isinstance(default_val, bool):
                        default_clause = f" DEFAULT {1 if default_val else 0}"
                    elif isinstance(default_val, (int, float)):
                        default_clause = f" DEFAULT {default_val}"
                    elif isinstance(default_val, str):
                        default_clause = f" DEFAULT '{default_val}'"
                conn.execute(text(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {sql_type}{default_clause}'
                ))
