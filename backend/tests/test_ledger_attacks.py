"""Classic ledger attacks (paper attack table, lower rows), on a throw-away ledger."""
from app import models
from app.ai import audit_ledger as al


def _chain(db, n=6):
    return [al.record_event(db, event_type="TEST_EVENT", payload={"i": i}, task_id=1) for i in range(n)]


def test_untouched_chain_is_valid(db):
    _chain(db)
    assert al.verify_full_chain(db)["status"] == "VALID"


def test_corrupted_signature_detected(db):
    ev = _chain(db)
    ev[2].signature = ("00" if ev[2].signature[:2] != "00" else "11") + ev[2].signature[2:]
    db.commit()
    assert al.verify_event(db, ev[2].event_id)["status"] == "SIGNATURE_INVALID"


def test_swapped_signature_detected(db):
    ev = _chain(db)
    ev[2].signature, ev[3].signature = ev[3].signature, ev[2].signature
    db.commit()
    assert al.verify_event(db, ev[2].event_id)["status"] == "SIGNATURE_INVALID"


def test_middle_deletion_detected(db):
    ev = _chain(db)
    db.delete(ev[2]); db.commit()
    assert al.verify_full_chain(db)["status"] == "CHAIN_BROKEN"
    assert al.verify_event(db, ev[4].event_id)["status"] == "CHAIN_BROKEN"


def test_replayed_copy_detected(db):
    ev = _chain(db)
    last = ev[-1]
    db.add(models.AuditEvent(event_id="copy" + last.event_id[4:], event_type=last.event_type,
                             payload_json=last.payload_json, actor_id=last.actor_id, task_id=last.task_id,
                             model_version=last.model_version, feature_schema_version=last.feature_schema_version,
                             rule_ids=last.rule_ids, previous_hash=last.previous_hash, event_hash=last.event_hash,
                             signature=last.signature, created_at=last.created_at))
    db.commit()
    assert al.verify_full_chain(db)["status"] == "CHAIN_BROKEN"


def test_tail_truncation_needs_batch_check(db):
    """Limitation: deleting the most recent events leaves a valid chain; the Merkle batch check catches it."""
    ev = _chain(db)
    batch = al.build_daily_batch(db, "2026-01-01")
    db.delete(ev[-1]); db.delete(ev[-2]); db.commit()
    assert al.verify_full_chain(db)["status"] == "VALID"
    assert al.verify_batch(db, batch["batch_id"])["status"] != "VALID"


def test_coordinated_rewrite_with_local_anchor_not_detected(db):
    """Limitation: truncation plus rewriting the batch row and the local stub anchor passes every check."""
    from app.ai import fabric_stub
    ev = _chain(db)
    batch = al.build_daily_batch(db, "2026-01-02")
    db.delete(ev[-1]); db.commit()
    remaining = db.query(models.AuditEvent).filter(models.AuditEvent.merkle_batch_id == batch["batch_id"]) \
        .order_by(models.AuditEvent.id).all()
    root = al._merkle_root([e.event_hash for e in remaining])
    b = db.query(models.MerkleBatch).get(batch["batch_id"])
    b.merkle_root, b.event_count = root, len(remaining)
    db.commit()
    fabric_stub.registerAuditRoot("2026-01-02", root, metadata_hash="x")
    assert al.verify_full_chain(db)["status"] == "VALID"
    assert al.verify_batch(db, batch["batch_id"])["status"] == "VALID"
