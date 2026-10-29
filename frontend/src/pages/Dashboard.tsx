import { useEffect, useState } from "react";
import { api } from "../api";
import ConfidenceRing from "../components/ConfidenceRing";
import { AIBadge } from "../components/Bits";

export default function Dashboard() {
  const [health, setHealth] = useState<any[]>([]);
  const [recs, setRecs] = useState<any[]>([]);
  const [workload, setWorkload] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.dashboardHealth(), api.recommendations("pending"), api.workload()])
      .then(([h, r, w]) => {
        setHealth(h);
        setRecs(r);
        setWorkload(w);
      })
      .finally(() => setLoading(false));
  }, []);

  const decide = async (id: number, decision: string) => {
    await api.decideRecommendation(id, decision);
    setRecs((prev) => prev.filter((r) => r.id !== id));
  };

  if (loading) return <div className="p-8 text-text-muted">Loading project intelligence…</div>;

  const overloaded = workload.filter((w) => w.overloaded);

  return (
    <div className="p-8 space-y-8 max-w-6xl">
      <div>
        <h1 className="font-display text-2xl font-semibold">Overview</h1>
        <p className="text-text-muted text-sm mt-1">Live project health, computed from real issue and sprint data.</p>
      </div>

      <section>
        <h2 className="font-display font-semibold text-sm text-text-muted uppercase tracking-wide mb-3">Project Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {health.map((p) => (
            <div key={p.project_id} className="card p-4">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="font-medium text-sm">{p.project_name}</div>
                  <div className="text-xs text-text-muted mt-0.5">{p.progress_pct}% complete</div>
                </div>
                <ConfidenceRing value={p.health_score} size={40} />
              </div>
              <div className="flex gap-4 text-xs text-text-muted mb-2">
                <span>Schedule {p.schedule_health}</span>
                <span>Risk {p.risk_health}</span>
                <span>Resource {p.resource_health}</span>
              </div>
              <div className="text-xs font-medium mb-1">
                Delay probability: <span className={p.delay_probability > 60 ? "text-danger" : "text-text"}>{p.delay_probability}%</span>
                {" · "}~{p.predicted_delay_days}d
              </div>
              <ul className="text-xs text-text-muted list-disc list-inside space-y-0.5 mt-2">
                {p.reasons.slice(0, 2).map((r: string, i: number) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section>
          <h2 className="font-display font-semibold text-sm text-text-muted uppercase tracking-wide mb-3">AI Recommendations</h2>
          <div className="space-y-3">
            {recs.length === 0 && <div className="card p-4 text-sm text-text-muted">No pending recommendations right now.</div>}
            {recs.map((r) => (
              <div key={r.id} className="card p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <AIBadge>{r.kind}</AIBadge>
                    <div className="font-medium text-sm mt-1.5">{r.title}</div>
                    <div className="text-xs text-text-muted mt-1">{r.reason}</div>
                  </div>
                  <ConfidenceRing value={r.confidence} size={36} />
                </div>
                <div className="flex gap-2 mt-3">
                  <button className="btn-primary !px-3 !py-1.5 text-xs" onClick={() => decide(r.id, "approved")}>Approve</button>
                  <button className="btn-ghost !px-3 !py-1.5 text-xs" onClick={() => decide(r.id, "rejected")}>Reject</button>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2 className="font-display font-semibold text-sm text-text-muted uppercase tracking-wide mb-3">Workload</h2>
          <div className="card p-4">
            {overloaded.length > 0 && (
              <div className="text-xs bg-danger/10 text-danger rounded-lg px-3 py-2 mb-3">
                {overloaded.length} team member(s) at or above 85% utilization.
              </div>
            )}
            <div className="space-y-2.5">
              {workload.slice(0, 8).map((w) => (
                <div key={w.employee_id} className="flex items-center gap-3">
                  <span className="text-xs w-28 truncate">{w.name}</span>
                  <div className="flex-1 h-2 bg-paper rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${w.overloaded ? "bg-danger" : w.underutilized ? "bg-warning" : "bg-brand"}`}
                      style={{ width: `${w.utilization}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-text-muted w-9 text-right">{w.utilization}%</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
