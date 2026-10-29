import { useEffect, useState } from "react";
import { api } from "../api";

export default function DecisionHistory() {
  const [decisions, setDecisions] = useState<any[]>([]);
  const [perf, setPerf] = useState<any>(null);

  useEffect(() => {
    api.decisions().then(setDecisions);
    api.modelPerformance().then(setPerf);
  }, []);

  return (
    <div className="p-8 max-w-5xl space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold">Decision History</h1>
        <p className="text-text-muted text-sm mt-1">Every AI action is logged with its model, confidence, and the human decision.</p>
      </div>

      {perf && (
        <div className="grid grid-cols-3 gap-4">
          <div className="card p-4">
            <div className="text-xs text-text-muted">Total decisions</div>
            <div className="text-2xl font-display font-semibold">{perf.total_decisions}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-text-muted">Acceptance rate</div>
            <div className="text-2xl font-display font-semibold">{perf.acceptance_rate}%</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-text-muted">Avg confidence</div>
            <div className="text-2xl font-display font-semibold">{perf.average_confidence}%</div>
          </div>
        </div>
      )}

      <div className="card divide-y divide-line">
        {decisions.length === 0 && <div className="p-4 text-sm text-text-muted">No decisions logged yet.</div>}
        {decisions.map((d) => (
          <div key={d.id} className="p-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium">{d.recommendation}</span>
              <span className="text-xs font-mono text-text-muted">{d.confidence}%</span>
            </div>
            <div className="text-xs text-text-muted flex items-center gap-3 flex-wrap">
              <span>{d.model}</span>
              <span>·</span>
              <span>{d.human_decision}</span>
              <span>·</span>
              <span>{new Date(d.created_at).toLocaleString()}</span>
            </div>
            {d.evidence && <div className="text-xs text-text-muted mt-1">{d.evidence}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
