import { useEffect, useState } from "react";
import { api } from "../api";
import ConfidenceRing from "../components/ConfidenceRing";
import { Avatar, AIBadge, RecoveryRiskBadge } from "../components/Bits";

export default function AbsenceRecovery() {
  const [employees, setEmployees] = useState<any[]>([]);
  const [selected, setSelected] = useState<number | "">("");
  const [stage, setStage] = useState<"idle" | "analyzing" | "done">("idle");
  const [results, setResults] = useState<any[]>([]);

  const load = () => api.employees().then(setEmployees);
  useEffect(() => { load(); }, []);

  const triggerAbsence = async () => {
    if (!selected) return;
    setStage("analyzing");
    setResults([]);
    await new Promise((r) => setTimeout(r, 700)); // let the pipeline steps feel visible
    const res = await api.markAbsent(Number(selected));
    setResults(res);
    setStage("done");
    load();
  };

  const approve = async (r: any) => {
    if (!r.recommended_employee_id) return;
    await api.approveReassignment(r.task_id, r.recommended_employee_id);
    setResults((prev) => prev.map((x) => (x.task_id === r.task_id ? { ...x, approved: true } : x)));
  };

  const returnEmployee = async () => {
    if (!selected) return;
    await api.returnFromAbsence(Number(selected));
    setSelected(""); setStage("idle"); setResults([]);
    load();
  };

  const availableEmployees = employees.filter((e) => e.status === "available");
  const absentEmployees = employees.filter((e) => e.status === "absent");

  return (
    <div className="p-8 max-w-5xl space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold">Absence-Aware Work Recovery</h1>
        <p className="text-text-muted text-sm mt-1">Mark someone unavailable and the system finds, ranks, and explains the best reassignment for every affected task.</p>
      </div>

      <div className="card p-5">
        <div className="flex items-end gap-3 flex-wrap">
          <div className="flex-1 min-w-[220px]">
            <label className="text-xs text-text-muted block mb-1">Mark employee absent</label>
            <select
              className="w-full border border-line rounded-lg px-3 py-2 text-sm outline-none focus:border-brand"
              value={selected}
              onChange={(e) => setSelected(e.target.value ? Number(e.target.value) : "")}
            >
              <option value="">Select employee…</option>
              {availableEmployees.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
          </div>
          <button className="btn-danger" disabled={!selected || stage === "analyzing"} onClick={triggerAbsence}>
            Mark Absent
          </button>
        </div>

        {absentEmployees.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {absentEmployees.map((e) => (
              <span key={e.id} className="badge bg-danger/10 text-danger">
                {e.name} — absent
                <button className="ml-1 underline" onClick={() => { setSelected(e.id); returnEmployee(); }}>mark returned</button>
              </span>
            ))}
          </div>
        )}
      </div>

      {stage === "analyzing" && (
        <div className="card p-6 text-sm text-text-muted space-y-2">
          <div className="flex items-center gap-2"><Spinner /> Detecting affected tasks…</div>
          <div className="flex items-center gap-2"><Spinner /> Analyzing skills, history and capacity across the team…</div>
          <div className="flex items-center gap-2"><Spinner /> Ranking candidates…</div>
        </div>
      )}

      {stage === "done" && (
        <div className="space-y-4">
          {results.map((r) => (
            <div key={r.task_id} className="card p-5">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="text-xs text-text-muted">Affected task</div>
                  <div className="font-medium">{r.task_title}</div>
                </div>
                <div className="flex items-center gap-2">
                  {r.confidence_label && <span className="badge bg-warning/10 text-warning">{r.confidence_label}</span>}
                  <AIBadge>{r.no_suitable_candidate ? "No Suitable Match" : "AI Recommended"}</AIBadge>
                </div>
              </div>

              {r.data_quality_flags && r.data_quality_flags.length > 0 && (
                <div className="bg-text-muted/10 text-text-muted text-xs rounded-lg p-2.5 mb-3">
                  <span className="font-medium">Data quality: </span>
                  {r.data_quality_flags.join(" · ")}
                </div>
              )}

              {r.no_suitable_candidate ? (
                <div className="bg-warning/10 text-warning text-sm rounded-lg p-3">
                  <div className="font-medium mb-1">No suitable reassignment found.</div>
                  <div>{r.reason}</div>
                  {r.rule_excluded_count > 0 && (
                    <div className="text-xs mt-2 opacity-80">
                      {r.rule_excluded_count} candidate(s) were excluded by policy rules — see reasoning below.
                    </div>
                  )}
                  <ul className="list-disc list-inside mt-2 text-xs">
                    {r.alternatives.map((a: string, i: number) => <li key={i}>{a}</li>)}
                  </ul>
                  {r.policy_explanation && r.policy_explanation.length > 0 && (
                    <ul className="list-disc list-inside mt-2 text-xs opacity-80">
                      {r.policy_explanation.map((p: string, i: number) => <li key={i}>{p}</li>)}
                    </ul>
                  )}
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-3 mb-3">
                    <Avatar name={r.recommended_employee_name} />
                    <div className="flex-1">
                      <div className="text-sm font-medium">{r.recommended_employee_name}</div>
                      <div className="text-xs text-text-muted">{r.reason}</div>
                    </div>
                    <ConfidenceRing value={r.confidence} size={44} label="Allocation score" />
                  </div>

                  {r.requires_manager_review && (
                    <div className="bg-warning/10 text-warning text-xs rounded-lg p-2.5 mb-3">
                      <span className="font-medium">Manager review required: </span>
                      {r.review_reasons?.join(" · ")}
                    </div>
                  )}

                  {(r.model_explanation || (r.policy_explanation && r.policy_explanation.length > 0)) && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3 text-xs">
                      {r.model_explanation && (
                        <div className="bg-paper rounded-lg p-2.5">
                          <div className="font-medium mb-1">Model explanation</div>
                          {r.model_explanation.top_positive_factors?.map((f: string, i: number) => (
                            <div key={i} className="text-success">+ {f}</div>
                          ))}
                          {r.model_explanation.top_negative_factors?.map((f: string, i: number) => (
                            <div key={i} className="text-danger">− {f}</div>
                          ))}
                        </div>
                      )}
                      {r.policy_explanation && r.policy_explanation.length > 0 && (
                        <div className="bg-paper rounded-lg p-2.5">
                          <div className="font-medium mb-1">Policy explanation (rules)</div>
                          {r.policy_explanation.map((p: string, i: number) => (
                            <div key={i} className="text-text-muted">{p}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                    {r.candidates.slice(0, 4).map((c: any) => (
                      <div key={c.employee_id} className={`bg-paper rounded-lg p-2.5 text-xs ${c.excluded_by_rule ? "opacity-50" : ""}`}>
                        <div className="font-medium truncate flex items-center gap-1">
                          {c.name}
                          {c.excluded_by_rule && <span className="text-danger text-[10px]">excluded</span>}
                        </div>
                        <div className="text-text-muted mt-1 space-y-0.5">
                          <div>Skill match: {c.skill}%</div>
                          <div>Success rate: {c.success_rate}%</div>
                          <div>Capacity: {c.capacity}%</div>
                          <div>Completion prob.: {c.completion_probability}%</div>
                          {c.workload_ratio !== undefined && <div>Load ratio: {c.workload_ratio}</div>}
                          {c.transfer_cost !== undefined && <div>Transfer cost: {c.transfer_cost}</div>}
                        </div>
                        {c.excluded_by_rule ? (
                          <div className="text-danger text-[11px] mt-1">{c.exclusion_reason}</div>
                        ) : c.recovery_risk_band ? (
                          <div className="mt-1.5"><RecoveryRiskBadge band={c.recovery_risk_band} /></div>
                        ) : null}
                      </div>
                    ))}
                  </div>

                  {r.approved ? (
                    <div className="text-sm text-success font-medium">✓ Reassigned and logged to Decision History</div>
                  ) : (
                    <div className="flex gap-2">
                      <button className="btn-primary" onClick={() => approve(r)}>Approve</button>
                      <button className="btn-ghost">Modify</button>
                      <button className="btn-ghost">Reject</button>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Spinner() {
  return <span className="inline-block w-3 h-3 border-2 border-brand border-t-transparent rounded-full animate-spin" />;
}
