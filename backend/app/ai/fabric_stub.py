"""
Hyperledger Fabric — OPTIONAL enterprise anchoring layer.

This is intentionally a stub, not a live Fabric network connection. A real
Fabric peer/orderer network isn't available in this sandbox (no Docker,
no way to stand up a channel), and the spec is explicit that the whole
application "must remain runnable without Hyperledger Fabric" — the SHA-256
hash chain + Ed25519 signatures + Merkle batching in audit_ledger.py
already provide the real integrity guarantee on their own.

What's here is the exact chaincode-shaped interface the spec asks for
(registerAuditRoot / getAuditRoot / verifyAnchor), backed by a local
append-only JSON file instead of a blockchain. This keeps the call sites
in audit_ledger.py identical to what they'd be if a real Fabric SDK client
were swapped in later — only this file would need to change, nowhere else.

Anchors here store ONLY: date, Merkle root, platform identifier, and a
metadata hash — never employee names, ticket text, or biometric data, per
the spec's requirement, matching what a real chaincode call would send.
"""
import json
import os
from datetime import datetime

LEDGER_STUB_PATH = os.path.join(os.path.dirname(__file__), "model_store", "fabric_stub_ledger.json")


def _load() -> dict:
    if not os.path.exists(LEDGER_STUB_PATH):
        return {}
    with open(LEDGER_STUB_PATH) as f:
        return json.load(f)


def _save(data: dict):
    os.makedirs(os.path.dirname(LEDGER_STUB_PATH), exist_ok=True)
    with open(LEDGER_STUB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def registerAuditRoot(batch_date: str, merkle_root: str, metadata_hash: str) -> dict:
    data = _load()
    tx_id = f"LOCAL-STUB-{batch_date}-{merkle_root[:12]}"
    data[batch_date] = {
        "merkle_root": merkle_root, "platform_identifier": "adaptiveflow-local",
        "metadata_hash": metadata_hash, "transaction_id": tx_id,
        "anchored_at": datetime.utcnow().isoformat() + "Z",
        "note": "Local stub anchor — not a real Hyperledger Fabric transaction. See fabric_stub.py.",
    }
    _save(data)
    return data[batch_date]


def getAuditRoot(batch_date: str) -> dict | None:
    return _load().get(batch_date)


def verifyAnchor(batch_date: str, merkle_root: str) -> bool:
    record = getAuditRoot(batch_date)
    return bool(record and record["merkle_root"] == merkle_root)
