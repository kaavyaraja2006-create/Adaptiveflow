import { useEffect, useState } from "react";
import { api } from "../api";

const SCENARIOS = [
  { type: "absence_extended", label: "Extend an absence" },
  { type: "add_developer", label: "Add a developer" },
  { type: "remove_feature", label: "Remove a feature" },
  { type: "change_methodology", label: "Switch methodology" },
  { type: "move_sprint", label: "Move to a later sprint" },
];

export default function Simulator() {
  const [projects, setProjects] = useState<any[]>([]);
  const [employees, setEmployees] = useState<any[]>([]);
  const [projectId, setProjectId] = useState<number | "">("");
  const [scenario, setScenario] = useState(SCENARIOS[0].type);
  const [days, setDays] = useState(3);
  const [employeeId, setEmployeeId] = useState<number | "">("");
  const [skillArea, setSkillArea] = useState("Backend");
  const [points, setPoints] = useState(8);
  const [sprints, setSprints] = useState(1);
  const [result, setResult] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [twin, setTwin] = useState<any>(null);
  const [twinRunning, setTwinRunning] = useState(false);

  useEffect(() => {
    api.projects().then((p) => { setProjects(p); if (p[0]) setProjectId(p[0].id); });
    api.employees().then(setEmployees);
  }, []);

  const paramsFor = () => {
    if (scenario === "absence_extended") return { employee_id: employeeId, days };
    if (scenario === "add_developer") return { skill_area: skillArea };
    if (scenario === "remove_feature") return { story_points: points };
    if (scenario === "move_sprint") return { sprints };
    return {};
  };

  const run = async () => {
    if (!projectId) return;
    setRunning(true);
    try {
      const r = await api.simulate(scenario, Number(projectId), paramsFor());
      setResult(r);
    } finally {
      setRunning(false);
    }
  };

  const runTwin = async () => {
    if (!projectId) return;
    setTwinRunning(true);
    try {
      const r = await api.simulatorRun(Number(projectId), undefined, 500);
      setTwin(r);
    } finally {
      setTwinRunning(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold">What-If Simulator</h1>
        <p className="text-text-muted text-sm mt-1">Test a hypothetical change and see its predicted effect, computed from the same engine as the live dashboard.</p>
      </div>

      <div className="card p-5 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-text-muted block mb-1">Project</label>
            <select className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={projectId} onChange={(e) => setProjectId(Number(e.target.value))}>
              {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-text-muted block mb-1">Scenario</label>
            <select className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {SCENARIOS.map((s) => <option key={s.type} value={s.type}>{s.label}</option>)}
            </select>
          </div>
        </div>

        {scenario === "absence_extended" && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-text-muted block mb-1">Employee</label>
              <select className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={employeeId} onChange={(e) => setEmployeeId(Number(e.target.value))}>
                <option value="">Select…</option>
                {employees.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-text-muted block mb-1">Extra days</label>
              <input type="number" className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={days} onChange={(e) => setDays(Number(e.target.value))} />
            </div>
          </div>
        )}
        {scenario === "add_developer" && (
          <div>
            <label className="text-xs text-text-muted block mb-1">Skill area</label>
            <select className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={skillArea} onChange={(e) => setSkillArea(e.target.value)}>
              {["Backend", "Frontend", "QA", "DevOps", "Mobile", "Security"].map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
        )}
        {scenario === "remove_feature" && (
          <div>
            <label className="text-xs text-text-muted block mb-1">Story points removed</label>
            <input type="number" className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={points} onChange={(e) => setPoints(Number(e.target.value))} />
          </div>
        )}
        {scenario === "move_sprint" && (
          <div>
            <label className="text-xs text-text-muted block mb-1">Sprints to delay</label>
            <input type="number" className="w-full border border-line rounded-lg px-3 py-2 text-sm" value={sprints} onChange={(e) => setSprints(Number(e.target.value))} />
          </div>
        )}

        <button className="btn-primary" disabled={running || !projectId} onClick={run}>
          {running ? "Simulating…" : "Run Simulation"}
        </button>
      </div>

      {result && <AnalysisCard result={result} />}

      <div className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-display text-lg font-semibold">Project Digital Twin</h2>
            <p className="text-text-muted text-xs mt-0.5">
              Monte Carlo simulation over the real task dependency graph — 500 iterations, sampling from
              each task's forecasted duration distribution.
            </p>
          </div>
          <button className="btn-primary shrink-0" disabled={twinRunning || !projectId} onClick={runTwin}>
            {twinRunning ? "Simulating…" : "Run Digital Twin"}
          </button>
        </div>

        {twin && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className={`rounded-xl2 px-4 py-3 text-center ${
                twin.sprint_risk_band === "Critical" ? "bg-danger/10 text-danger" :
                twin.sprint_risk_band === "High" ? "bg-warning/20 text-warning" :
                twin.sprint_risk_band === "Moderate" ? "bg-warning/10 text-warning" : "bg-success/10 text-success"
              }`}>
                <div className="font-display text-2xl font-bold">{Math.round(twin.sprint_risk * 100)}%</div>
                <div className="text-xs mt-0.5">Sprint Risk — {twin.sprint_risk_band}</div>
              </div>
              <div className="flex-1 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                {Object.entries(twin.components).map(([k, v]: any) => (
                  <div key={k} className="bg-paper rounded-lg p-2">
                    <div className="text-text-muted">{k.replace(/_/g, " ")}</div>
                    <div className="font-mono font-semibold">{Math.round(v * 100)}%</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">Sprint miss probability</div>
                <div className="font-mono font-semibold">{Math.round(twin.simulation.sprint_miss_probability * 100)}%</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">p50 completion</div>
                <div className="font-mono font-semibold">{twin.simulation.p50_hours}h</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">p80 completion</div>
                <div className="font-mono font-semibold">{twin.simulation.p80_hours}h</div>
              </div>
              <div className="bg-paper rounded-lg p-2.5">
                <div className="text-text-muted">p95 completion</div>
                <div className="font-mono font-semibold">{twin.simulation.p95_hours}h</div>
              </div>
            </div>

            {twin.simulation.critical_tasks?.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-muted uppercase tracking-wide mb-2">Critical tasks</div>
                <div className="space-y-1">
                  {twin.simulation.critical_tasks.map((t: any) => (
                    <div key={t.issue_id} className="flex items-center justify-between text-xs bg-paper rounded-lg px-3 py-2">
                      <span>{t.title}</span>
                      <span className="text-text-muted">on critical path in {Math.round(t.critical_path_frequency * 100)}% of runs</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {twin.triggers?.length > 0 && (
              <div className="bg-warning/10 text-warning text-xs rounded-lg p-3 space-y-1">
                {twin.triggers.map((t: any, i: number) => <div key={i}>⚠ {t.message}</div>)}
              </div>
            )}

            <div className="text-[11px] text-text-muted">
              Data quality: {twin.data_quality.data_health_pct}% healthy across {twin.data_quality.checked_issues} issues
              and {twin.data_quality.checked_employees} employees.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function AnalysisCard({ result }: { result: any }) {
  return (
    <div className="rounded-2xl overflow-hidden border border-line shadow-sm">
      <div className="bg-ink text-white px-6 py-4 flex items-center gap-2">
        <span className="text-lg">🤖</span>
        <span className="font-display font-semibold text-sm tracking-wide">{result.title}</span>
      </div>

      <div className="bg-surface px-6 py-6 space-y-6">
        <h2 className="font-display text-xl font-semibold leading-snug">{result.headline}</h2>

        {result.compare?.length > 0 && (
          <div className="grid grid-cols-2 gap-4">
            {result.compare.map((c: any, i: number) => (
              <CompareBox key={i} {...c} />
            ))}
          </div>
        )}

        {result.why?.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wide mb-3">Why?</div>
            <div className="flex gap-6 flex-wrap">
              {result.why.map((w: any, i: number) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xl">{w.icon}</span>
                  <div>
                    <div className="font-mono font-semibold text-sm">{w.value}</div>
                    <div className="text-xs text-text-muted">{w.label}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="border-t border-line" />

        <div className="bg-brand-dim rounded-xl2 p-4">
          <div className="text-xs font-semibold text-brand-deep uppercase tracking-wide mb-1.5 flex items-center gap-1.5">
            💡 AI Insight
          </div>
          <p className="text-sm text-text">{result.insight}</p>
        </div>

        <div className="flex items-center justify-between bg-ink rounded-xl2 px-5 py-4">
          <div className="flex items-center gap-2">
            <span>🎯</span>
            <div>
              <div className="text-xs text-white/50 uppercase tracking-wide">Recommendation</div>
              <div className="text-white font-display font-semibold">{result.recommendation.label}</div>
            </div>
          </div>
          <div className="text-white font-mono font-semibold text-lg">{result.recommendation.confidence}%</div>
        </div>
      </div>
    </div>
  );
}

function CompareBox({ label, value, verdict, tone, recommended }: any) {
  const toneColor = tone === "good" ? "text-success" : tone === "bad" ? "text-danger" : "text-text";
  const borderColor = recommended ? "border-brand" : "border-line";
  return (
    <div className={`relative rounded-xl2 border-2 ${borderColor} p-4 text-center bg-paper`}>
      {recommended && (
        <span className="absolute top-2 right-2 text-success text-sm">✓</span>
      )}
      <div className="text-xs font-semibold text-text-muted uppercase tracking-wide mb-2">{label}</div>
      <div className={`font-display text-3xl font-bold ${toneColor}`}>{value}%</div>
      <div className="text-xs text-text-muted mt-1">{verdict}</div>
    </div>
  );
}
