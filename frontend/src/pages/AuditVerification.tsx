import { useEffect, useState } from "react";
import { api } from "../api";
import { AIBadge } from "../components/Bits";

const STATUS_STYLE: Record<string, string> = {
  VALID: "bg-success/10 text-success",
  CHAIN_BROKEN: "bg-danger/10 text-danger",
  SIGNATURE_INVALID: "bg-danger/10 text-danger",
  ANCHOR_NOT_FOUND: "bg-warning/10 text-warning",
  VERIFICATION_ERROR: "bg-warning/10 text-warning",
};

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge text-sm px-3 py-1 ${STATUS_STYLE[status] || "bg-text-muted/10 text-text-muted"}`}>
      {status.replace(/_/g, " ")}
    </span>
  );
}

export default function AuditVerification() {
  const [taskId, setTaskId] = useState("");
  const [trail, setTrail] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [verification, setVerification] = useState<any>(null);
  const [chainStatus, setChainStatus] = useState<any>(null);
  const [batches, setBatches] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const loadChain = () => api.verifyAuditChain().then(setChainStatus);
  const loadBatches = () => api.listAuditBatches().then(setBatches);
  useEffect(() => { loadChain(); loadBatches(); }, []);

  const lookupTask = async () => {
    if (!taskId) return;
    setLoading(true);
    setSelected(null);
    setVerification(null);
    try {
      const events = await api.taskAuditTrail(Number(taskId));
      setTrail(events);
    } finally {
      setLoading(false);
    }
  };

  const verify = async (eventId: string) => {
    const ev = trail.find((e) => e.event_id === eventId);
    setSelected(ev);
    setVerification(null);
    const result = await api.verifyAuditEvent(eventId);
    setVerification(result);
  };

  const anchorDaily = async () => {
    await api.anchorDailyBatch();
    loadBatches();
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="font-display text-2xl font-semibold">Audit Verification</h1>
        <p className="text-text-muted text-sm mt-1">
          Every AI recommendation, manager decision, and reassignment is recorded in a SHA-256 hash-chained,
          Ed25519-signed ledger. Look up a task to inspect its full decision trail and verify integrity.
        </p>
      </div>

      <div className="card p-5 mb-4 flex items-center justify-between">
        <div>
          <div className="text-sm font-medium">Full chain integrity</div>
          <div className="text-xs text-text-muted mt-0.5">
            {chainStatus ? `${chainStatus.events_checked} event(s) checked` : "Checking…"}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {chainStatus && <StatusBadge status={chainStatus.status} />}
          <button className="btn-ghost text-xs" onClick={loadChain}>Re-check</button>
        </div>
      </div>

      <div className="card p-5 mb-4">
        <div className="flex items-center gap-2">
          <input
            className="input flex-1"
            placeholder="Task / issue ID"
            value={taskId}
            onChange={(e) => setTaskId(e.target.value)}
          />
          <button className="btn-primary" onClick={lookupTask} disabled={loading}>
            {loading ? "Loading…" : "Look up"}
          </button>
        </div>

        {trail.length > 0 && (
          <div className="mt-4 space-y-2">
            {trail.map((e) => (
              <button
                key={e.event_id}
                onClick={() => verify(e.event_id)}
                className={`w-full text-left flex items-center justify-between text-xs bg-paper rounded-lg px-3 py-2 hover:bg-line/40 transition-colors ${selected?.event_id === e.event_id ? "ring-2 ring-brand" : ""}`}
              >
                <span className="font-medium">{e.event_type.replace(/_/g, " ")}</span>
                <span className="text-text-muted font-mono">{e.event_hash.slice(0, 12)}…</span>
              </button>
            ))}
          </div>
        )}
        {trail.length === 0 && taskId && !loading && (
          <p className="text-xs text-text-muted mt-3">No audit events found for this task yet.</p>
        )}
      </div>

      {verification && (
        <div className="card p-5 mb-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-display text-lg font-semibold">Verification result</h2>
            <StatusBadge status={verification.status} />
          </div>
          {verification.reason && <p className="text-xs text-text-muted mb-3">{verification.reason}</p>}
          {verification.status === "VALID" && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-paper rounded-lg p-2.5 col-span-2">
                <div className="text-text-muted">Event hash</div>
                <div className="font-mono break-all">{verification.event_hash}</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5 col-span-2">
                <div className="text-text-muted">Previous hash</div>
                <div className="font-mono break-all">{verification.previous_hash || "GENESIS"}</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5 col-span-2">
                <div className="text-text-muted">Signature (Ed25519)</div>
                <div className="font-mono break-all">{verification.signature?.slice(0, 64)}…</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">Model version</div>
                <div className="font-mono">{verification.model_version || "—"}</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">Rules applied</div>
                <div className="font-mono">{verification.rule_ids?.join(", ") || "—"}</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5 col-span-2">
                <div className="text-text-muted">Anchor status</div>
                <div className="font-mono">{verification.anchor_status}</div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="font-display text-lg font-semibold">Merkle batches</h2>
            <p className="text-xs text-text-muted mt-0.5">Daily-batched anchor roots (local stub — see Anthropic docs for enterprise Fabric anchoring).</p>
          </div>
          <button className="btn-ghost text-xs" onClick={anchorDaily}>Anchor unbatched events now</button>
        </div>
        <div className="space-y-2">
          {batches.map((b) => (
            <div key={b.id} className="flex items-center justify-between text-xs bg-paper rounded-lg px-3 py-2">
              <span>{b.batch_date} · {b.event_count} events</span>
              <span className="font-mono text-text-muted">{b.merkle_root.slice(0, 16)}…</span>
            </div>
          ))}
          {batches.length === 0 && <p className="text-xs text-text-muted">No batches anchored yet.</p>}
        </div>
      </div>
    </div>
  );
}
