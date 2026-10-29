"""
Immutable AI Decision Ledger — Phase 8-12.

A cryptographically chained, append-only audit ledger. Blockchain is
deliberately NOT the starting point (see fabric_stub.py): the real
integrity guarantee here is the SHA-256 hash chain + Ed25519 signature,
which works completely standalone. Daily Merkle batching lets a batch of
events be anchored with ONE root hash instead of one transaction per
event, if/when an enterprise deployment wants to plug in a real anchor
(Hyperledger Fabric or otherwise).

Nothing sensitive (employee names, ticket text, biometric data) is put in
the payload beyond what's already visible elsewhere in the app via normal
authenticated access — this ledger records DECISION PROVENANCE (what was
recommended, on what basis, what a manager decided, what happened), not a
duplicate copy of private HR data.
"""
import hashlib
import json
import os
import uuid
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy.orm import Session
from .. import models

KEY_DIR = os.path.join(os.path.dirname(__file__), "model_store")
PRIVATE_KEY_PATH = os.path.join(KEY_DIR, "audit_signing_key.pem")
PUBLIC_KEY_META_PATH = os.path.join(KEY_DIR, "audit_public_key.json")


def canonical_json(payload: dict) -> str:
    """json.dumps(payload, sort_keys=True, separators=(",", ":")) — the
    exact canonicalization the spec calls for, so independent
    re-verification of a hash is reproducible byte-for-byte."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_event_hash(payload: dict, previous_hash: Optional[str]) -> str:
    canonical = canonical_json(payload)
    material = canonical + (previous_hash or "GENESIS")
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- Ed25519 --

def _get_or_create_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    os.makedirs(KEY_DIR, exist_ok=True)
    if os.path.exists(PRIVATE_KEY_PATH):
        with open(PRIVATE_KEY_PATH, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
    else:
        private_key = Ed25519PrivateKey.generate()
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(pem)
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        with open(PUBLIC_KEY_META_PATH, "w") as f:
            json.dump({"public_key_hex": public_bytes.hex(), "key_version": "v1",
                       "created_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    return private_key


def get_public_key_info() -> dict:
    _get_or_create_keypair()  # ensures it exists
    if os.path.exists(PUBLIC_KEY_META_PATH):
        with open(PUBLIC_KEY_META_PATH) as f:
            return json.load(f)
    # key existed already (loaded from disk) but meta file is missing — rebuild it
    from cryptography.hazmat.primitives import serialization
    private_key = _get_or_create_keypair()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return {"public_key_hex": public_bytes.hex(), "key_version": "v1"}


def sign_hash(event_hash_hex: str) -> str:
    private_key = _get_or_create_keypair()
    signature = private_key.sign(bytes.fromhex(event_hash_hex))
    return signature.hex()


def verify_signature(event_hash_hex: str, signature_hex: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    info = get_public_key_info()
    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(info["public_key_hex"]))
    try:
        public_key.verify(bytes.fromhex(signature_hex), bytes.fromhex(event_hash_hex))
        return True
    except InvalidSignature:
        return False


# ------------------------------------------------------------- recording --

def record_event(
    db: Session, *, event_type: str, payload: dict, actor_id: str = "system-ai",
    project_id: Optional[int] = None, task_id: Optional[int] = None,
    model_version: Optional[str] = None, feature_schema_version: Optional[str] = "v1",
    rule_ids: Optional[List[str]] = None,
) -> models.AuditEvent:
    last = db.query(models.AuditEvent).order_by(models.AuditEvent.id.desc()).first()
    previous_hash = last.event_hash if last else None

    recorded_at = datetime.utcnow()
    full_payload = {
        "event_type": event_type, "actor_id": actor_id, "project_id": project_id,
        "task_id": task_id, "model_version": model_version,
        "feature_schema_version": feature_schema_version, "rule_ids": rule_ids or [],
        # the timestamp is now inside the hashed payload, and verify_event checks the
        # queryable columns below against these hashed copies (closes paper test T6)
        "recorded_at": recorded_at.isoformat(),
        "data": payload,
    }
    event_hash = compute_event_hash(full_payload, previous_hash)
    signature = sign_hash(event_hash)

    row = models.AuditEvent(
        event_id=uuid.uuid4().hex, event_type=event_type, payload_json=full_payload,
        actor_id=actor_id, project_id=project_id, task_id=task_id,
        model_version=model_version, feature_schema_version=feature_schema_version,
        rule_ids=rule_ids or [], previous_hash=previous_hash, event_hash=event_hash,
        signature=signature, created_at=recorded_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ------------------------------------------------------------- verifying --

def verify_event(db: Session, event_id: str) -> dict:
    event = db.query(models.AuditEvent).filter(models.AuditEvent.event_id == event_id).first()
    if not event:
        return {"status": "VERIFICATION_ERROR", "reason": "Event not found"}

    recomputed_hash = compute_event_hash(event.payload_json, event.previous_hash)
    if recomputed_hash != event.event_hash:
        return {"status": "CHAIN_BROKEN", "reason": "Stored payload no longer matches its recorded hash — "
                "the event has been altered since it was written.", "event_id": event_id}

    if not verify_signature(event.event_hash, event.signature):
        return {"status": "SIGNATURE_INVALID", "reason": "Ed25519 signature does not match the event hash.",
                "event_id": event_id}

    altered = column_mismatches(event)
    if altered:
        return {"status": "COLUMNS_ALTERED", "event_id": event_id, "fields": altered,
                "reason": "Queryable columns no longer match the hashed copy in the payload."}

    # walk the chain backward from this event to genesis, verifying every link
    chain_status = _verify_chain_up_to(db, event)
    if chain_status["status"] != "VALID":
        return chain_status

    anchor_status = "NOT_BATCHED"
    if event.merkle_batch_id:
        batch = db.query(models.MerkleBatch).get(event.merkle_batch_id)
        anchor_status = "ANCHORED" if batch else "ANCHOR_NOT_FOUND"

    return {
        "status": "VALID", "event_id": event_id, "event_type": event.event_type,
        "event_hash": event.event_hash, "previous_hash": event.previous_hash,
        "signature": event.signature, "model_version": event.model_version,
        "rule_ids": event.rule_ids, "created_at": event.created_at,
        "anchor_status": anchor_status, "merkle_batch_id": event.merkle_batch_id,
    }


ENVELOPE_FIELDS = ["event_type", "actor_id", "project_id", "task_id", "model_version",
                   "feature_schema_version", "rule_ids"]


def column_mismatches(event: models.AuditEvent) -> List[str]:
    """Columns that disagree with their hashed copies inside payload_json."""
    p = event.payload_json or {}
    bad = [f for f in ENVELOPE_FIELDS if f in p and getattr(event, f) != p[f]]
    if "recorded_at" in p and event.created_at is not None and event.created_at.isoformat() != p["recorded_at"]:
        bad.append("created_at")
    return bad


def latest_event(db: Session, task_id: int, event_types: List[str]) -> Optional[models.AuditEvent]:
    return (db.query(models.AuditEvent)
            .filter(models.AuditEvent.task_id == task_id, models.AuditEvent.event_type.in_(event_types))
            .order_by(models.AuditEvent.id.desc()).first())


def _verify_chain_up_to(db: Session, event: models.AuditEvent) -> dict:
    """Recompute hashes for this event and every ancestor back to genesis."""
    current = event
    checked = 0
    while current is not None:
        recomputed = compute_event_hash(current.payload_json, current.previous_hash)
        if recomputed != current.event_hash:
            return {"status": "CHAIN_BROKEN",
                    "reason": f"Chain link broken at event {current.event_id} (position {checked} back from target).",
                    "event_id": event.event_id}
        checked += 1
        if current.previous_hash is None:
            break
        current = db.query(models.AuditEvent).filter(models.AuditEvent.event_hash == current.previous_hash).first()
        if current is None:
            return {"status": "CHAIN_BROKEN",
                    "reason": "A previous_hash reference doesn't match any stored event — the chain is broken "
                              "or an earlier event is missing.", "event_id": event.event_id}
        if checked > 100000:
            break  # safety guard against pathological loops; not expected in a real chain
    return {"status": "VALID"}


def verify_full_chain(db: Session) -> dict:
    events = db.query(models.AuditEvent).order_by(models.AuditEvent.id.asc()).all()
    if not events:
        return {"status": "VALID", "events_checked": 0, "note": "Ledger is empty."}
    previous_hash = None
    for e in events:
        recomputed = compute_event_hash(e.payload_json, e.previous_hash)
        if recomputed != e.event_hash or e.previous_hash != previous_hash:
            return {"status": "CHAIN_BROKEN", "broken_at_event_id": e.event_id, "events_checked": events.index(e)}
        if not verify_signature(e.event_hash, e.signature):
            return {"status": "SIGNATURE_INVALID", "broken_at_event_id": e.event_id, "events_checked": events.index(e)}
        altered = column_mismatches(e)
        if altered:
            return {"status": "COLUMNS_ALTERED", "broken_at_event_id": e.event_id, "fields": altered,
                    "events_checked": events.index(e)}
        previous_hash = e.event_hash
    return {"status": "VALID", "events_checked": len(events)}


# --------------------------------------------------------------- Merkle --

def _merkle_root(hashes: List[str]) -> str:
    if not hashes:
        return hashlib.sha256(b"EMPTY").hexdigest()
    level = list(hashes)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # standard duplicate-last-node rule for odd counts
        level = [hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest()
                 for i in range(0, len(level), 2)]
    return level[0]


def build_daily_batch(db: Session, batch_date: Optional[str] = None) -> dict:
    """Batches every currently-unbatched AuditEvent into one Merkle tree.
    In a real deployment this would run once per day (see the trigger
    system); for the demo it can be called on demand."""
    batch_date = batch_date or date.today().isoformat()
    unbatched = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.merkle_batch_id.is_(None))
        .order_by(models.AuditEvent.id.asc())
        .all()
    )
    if not unbatched:
        return {"status": "no_events", "batch_date": batch_date}

    hashes = [e.event_hash for e in unbatched]
    root = _merkle_root(hashes)

    existing = db.query(models.MerkleBatch).filter(models.MerkleBatch.batch_date == batch_date).first()
    if existing:
        # extend the batch with today's newly-unbatched events (recompute root over ALL of them)
        prior_events = db.query(models.AuditEvent).filter(models.AuditEvent.merkle_batch_id == existing.id).all()
        hashes = [e.event_hash for e in prior_events] + hashes
        root = _merkle_root(hashes)
        existing.merkle_root = root
        existing.event_count = len(hashes)
        batch = existing
    else:
        batch = models.MerkleBatch(batch_date=batch_date, merkle_root=root, event_count=len(hashes))
        db.add(batch)
        db.flush()

    for e in unbatched:
        e.merkle_batch_id = batch.id
    db.commit()
    db.refresh(batch)

    from . import fabric_stub
    anchor = fabric_stub.registerAuditRoot(batch_date, root, metadata_hash=hashlib.sha256(
        f"adaptiveflow-local:{batch_date}".encode()).hexdigest())
    batch.blockchain_tx_id = anchor["transaction_id"]
    db.commit()

    return {"status": "batched", "batch_id": batch.id, "batch_date": batch_date,
            "merkle_root": root, "event_count": batch.event_count, "anchor": anchor}


def verify_batch(db: Session, batch_id: int) -> dict:
    batch = db.query(models.MerkleBatch).get(batch_id)
    if not batch:
        return {"status": "ANCHOR_NOT_FOUND", "reason": "No such batch."}
    events = (
        db.query(models.AuditEvent)
        .filter(models.AuditEvent.merkle_batch_id == batch_id)
        .order_by(models.AuditEvent.id.asc())
        .all()
    )
    recomputed = _merkle_root([e.event_hash for e in events])
    if recomputed != batch.merkle_root:
        return {"status": "CHAIN_BROKEN", "reason": "Recomputed Merkle root does not match the stored root — "
                "an event in this batch was altered or removed.", "batch_id": batch_id}

    from . import fabric_stub
    anchor_ok = fabric_stub.verifyAnchor(batch.batch_date, batch.merkle_root)
    return {"status": "VALID" if anchor_ok else "ANCHOR_NOT_FOUND", "batch_id": batch_id,
            "merkle_root": batch.merkle_root, "event_count": batch.event_count,
            "blockchain_tx_id": batch.blockchain_tx_id}
