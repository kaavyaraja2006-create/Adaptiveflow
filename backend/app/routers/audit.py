"""
Phase 8-13 endpoints: audit event lookup, cryptographic verification, and
daily Merkle anchoring.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models
from ..database import get_db
from ..ai import audit_ledger

router = APIRouter(prefix="/api/audit", tags=["audit"])


def _serialize(e: models.AuditEvent) -> dict:
    return {
        "event_id": e.event_id, "event_type": e.event_type, "actor_id": e.actor_id,
        "project_id": e.project_id, "task_id": e.task_id, "model_version": e.model_version,
        "feature_schema_version": e.feature_schema_version, "rule_ids": e.rule_ids,
        "payload": e.payload_json, "previous_hash": e.previous_hash, "event_hash": e.event_hash,
        "signature": e.signature, "merkle_batch_id": e.merkle_batch_id, "created_at": e.created_at,
    }


@router.get("/events/{event_id}")
def get_event(event_id: str, db: Session = Depends(get_db)):
    e = db.query(models.AuditEvent).filter(models.AuditEvent.event_id == event_id).first()
    if not e:
        raise HTTPException(404, "Event not found")
    return _serialize(e)


@router.get("/tasks/{task_id}")
def get_task_audit_trail(task_id: int, db: Session = Depends(get_db)):
    events = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.task_id == task_id)
        .order_by(models.AuditEvent.id.asc())
        .all()
    )
    return [_serialize(e) for e in events]


@router.get("/projects/{project_id}")
def get_project_audit_trail(project_id: int, db: Session = Depends(get_db)):
    events = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.project_id == project_id)
        .order_by(models.AuditEvent.id.desc())
        .limit(100)
        .all()
    )
    return [_serialize(e) for e in events]


@router.get("/verify/event/{event_id}")
def verify_event(event_id: str, db: Session = Depends(get_db)):
    return audit_ledger.verify_event(db, event_id)


@router.get("/verify/batch/{batch_id}")
def verify_batch(batch_id: int, db: Session = Depends(get_db)):
    return audit_ledger.verify_batch(db, batch_id)


@router.get("/verify/decision/{event_id}")
def verify_decision(event_id: str, db: Session = Depends(get_db)):
    """Ledger integrity + offline replay of the decision receipt carried by the event."""
    from ..ai import decision_receipt
    return decision_receipt.verify_decision(db, event_id)


@router.get("/verify/chain")
def verify_full_chain(db: Session = Depends(get_db)):
    return audit_ledger.verify_full_chain(db)


@router.post("/anchor/daily")
def anchor_daily(db: Session = Depends(get_db)):
    return audit_ledger.build_daily_batch(db)


@router.get("/batches")
def list_batches(db: Session = Depends(get_db)):
    batches = db.query(models.MerkleBatch).order_by(models.MerkleBatch.id.desc()).limit(30).all()
    return [
        {"id": b.id, "batch_date": b.batch_date, "merkle_root": b.merkle_root,
         "event_count": b.event_count, "blockchain_tx_id": b.blockchain_tx_id,
         "created_at": b.created_at}
        for b in batches
    ]


@router.get("/public-key")
def public_key():
    """The Ed25519 public key any independent verifier would need to
    confirm a signature — this is safe and meant to be shared; it's the
    private key (never exposed via the API) that must stay secret."""
    return audit_ledger.get_public_key_info()
