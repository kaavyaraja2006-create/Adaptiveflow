import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Avatar, RecoveryRiskBadge } from "../components/Bits";
import { ensureModelsLoaded, startWebcam, stopWebcam, captureDescriptor } from "../face";

export default function Team() {
  const [employees, setEmployees] = useState<any[]>([]);
  const [teams, setTeams] = useState<any[]>([]);
  const [profile, setProfile] = useState<any>(null);
  const [workload, setWorkload] = useState<any>(null);
  const [enrolling, setEnrolling] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState<any>(null);
  const [training, setTraining] = useState(false);

  const load = () => api.employees().then(setEmployees);
  const loadModelStatus = () => api.sequenceModelStatus().then(setModelStatus);
  useEffect(() => { load(); api.teams().then(setTeams); loadModelStatus(); }, []);

  const trainModel = async () => {
    setTraining(true);
    try {
      const result = await api.trainSequenceModel();
      setModelStatus(result);
      if (profile) {
        const w = await api.workloadProfile(profile.employee_id);
        setWorkload(w);
      }
    } finally {
      setTraining(false);
    }
  };

  const openProfile = async (id: number) => {
    const p = await api.employeeProfile(id);
    setProfile(p);
    setWorkload(null);
    try {
      const w = await api.workloadProfile(id);
      setWorkload(w);
    } catch {
      setWorkload(null);
    }
  };

  const toggleOptOut = async () => {
    if (!profile) return;
    const next = !workload?.opted_out;
    await api.setWorkloadOptOut(profile.employee_id, next);
    const w = await api.workloadProfile(profile.employee_id);
    setWorkload(w);
    load();
  };

  const teamName = (id: number) => teams.find((t) => t.id === id)?.name || "—";

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold">Team</h1>
          <p className="text-text-muted text-sm mt-1">Capability profiles computed from real work history — click anyone to see the breakdown.</p>
        </div>
        <div className="card px-4 py-3 text-xs shrink-0 min-w-[240px]">
          <div className="font-medium mb-1">Recovery-risk sequence model</div>
          {modelStatus?.status === "trained" ? (
            <div className="text-text-muted">
              <span className="text-success font-medium">Trained</span> · {modelStatus.n_samples} sequence windows ·
              val loss {modelStatus.final_val_risk_mse}
            </div>
          ) : modelStatus?.status === "insufficient_data" ? (
            <div className="text-text-muted">
              <span className="text-warning font-medium">Insufficient data</span> — {modelStatus.n_samples}/{modelStatus.min_required} windows available
            </div>
          ) : (
            <div className="text-text-muted">Not trained yet — using the deterministic baseline</div>
          )}
          <button className="btn-ghost mt-2 w-full text-xs" onClick={trainModel} disabled={training}>
            {training ? "Training…" : "Train BiGRU on seeded data"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {employees.map((e) => (
          <div key={e.id} className="card p-4 hover:border-brand transition-colors">
            <button onClick={() => openProfile(e.id)} className="w-full text-left">
              <div className="flex items-center gap-3 mb-2">
                <Avatar name={e.name} color={e.avatar_color} />
                <div>
                  <div className="text-sm font-medium">{e.name}</div>
                  <div className="text-xs text-text-muted">{e.role} · {teamName(e.team_id)} · {e.office_id}</div>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className={e.status === "absent" ? "text-danger" : "text-success"}>{e.status}</span>
                <span className="text-text-muted font-mono">{Math.round(e.utilization)}% utilized</span>
              </div>
            </button>
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-line text-xs">
              <span className={e.face_enrolled ? "text-success" : "text-text-muted"}>
                {e.face_enrolled ? "✓ Face enrolled" : "No face enrolled"}
              </span>
              <button className="underline text-brand" onClick={() => setEnrolling(e)}>
                {e.face_enrolled ? "Re-enroll" : "Enroll face"}
              </button>
            </div>
          </div>
        ))}
      </div>

      {profile && (
        <div className="card p-5 max-w-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display font-semibold">{profile.name}</h2>
            <button className="text-xs text-text-muted underline" onClick={() => setProfile(null)}>close</button>
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm mb-4">
            <div><div className="text-xs text-text-muted">Avg success</div>{profile.avg_success_rate}%</div>
            <div><div className="text-xs text-text-muted">Completed</div>{profile.total_completed} tasks</div>
            <div><div className="text-xs text-text-muted">Capacity</div>{profile.available_capacity}% free</div>
          </div>
          <div className="space-y-2">
            {Object.entries(profile.breakdown).map(([type, b]: any) => (
              <div key={type} className="flex items-center gap-3 text-xs">
                <span className="w-20 shrink-0">{type}</span>
                <div className="flex-1 h-2 bg-paper rounded-full overflow-hidden">
                  <div className="h-full bg-brand rounded-full" style={{ width: `${b.skill}%` }} />
                </div>
                <span className="font-mono text-text-muted w-32 shrink-0">
                  {b.skill}% skill · {b.success_rate}% success ({b.completed_tasks})
                </span>
              </div>
            ))}
          </div>

          <div className="mt-5 pt-4 border-t border-line">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium">Sustainable Workload / Recovery Risk</h3>
              <button className="text-xs text-brand underline" onClick={toggleOptOut}>
                {workload?.opted_out ? "Opt back in" : "Opt out"}
              </button>
            </div>
            {!workload ? (
              <p className="text-xs text-text-muted">Loading…</p>
            ) : workload.opted_out ? (
              <p className="text-xs text-text-muted">
                This employee has opted out of workload/recovery-risk analysis. No workload data is scored
                or shown for them anywhere in AdaptiveFlow.
              </p>
            ) : (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <RecoveryRiskBadge band={workload.risk_band} />
                  <span className="text-xs text-text-muted">{workload.model_status.replace(/_/g, " ")}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-text-muted">
                  <div>Load ratio: <span className="font-mono">{workload.load_ratio}</span></div>
                  <div>After-hours ratio: <span className="font-mono">{workload.after_hours_ratio}</span></div>
                  <div>Context switching: <span className="font-mono">{workload.context_switching}</span></div>
                  <div>Rest days (7d): <span className="font-mono">{workload.rest_days_recent}</span></div>
                  <div>Expected velocity (7d): <span className="font-mono">{workload.expected_velocity_next_7_days}</span></div>
                  <div>Prediction confidence: <span className="font-mono">{Math.round(workload.prediction_confidence * 100)}%</span></div>
                </div>
                {workload.stale_data_flags?.length > 0 && (
                  <p className="text-xs text-warning mt-2">{workload.stale_data_flags.join(" · ")}</p>
                )}
                <p className="text-[11px] text-text-muted mt-3 italic">
                  A work-pattern signal only — never a medical or emotional assessment.
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {enrolling && (
        <EnrollModal employee={enrolling} onClose={() => setEnrolling(null)} onDone={() => { setEnrolling(null); load(); }} />
      )}
    </div>
  );
}

function EnrollModal({ employee, onClose, onDone }: { employee: any; onClose: () => void; onDone: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Loading face recognition models…");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureModelsLoaded();
      if (cancelled) return;
      setReady(true);
      setStatus("");
      if (videoRef.current) streamRef.current = await startWebcam(videoRef.current);
    })();
    return () => { cancelled = true; stopWebcam(streamRef.current); };
  }, []);

  const save = async () => {
    if (!videoRef.current) return;
    setSaving(true);
    setStatus("Capturing…");
    try {
      const descriptor = await captureDescriptor(videoRef.current);
      if (!descriptor) {
        setStatus("No face detected — center your face in frame and try again.");
        return;
      }
      await api.enrollFace(employee.id, descriptor);
      onDone();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" onClick={onClose}>
      <div className="card p-5 w-full max-w-sm" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display font-semibold">Enroll face — {employee.name}</h3>
          <button className="text-xs text-text-muted underline" onClick={onClose}>close</button>
        </div>
        <video ref={videoRef} className="rounded-lg bg-ink w-full mb-3" muted playsInline />
        <button className="btn-primary w-full" disabled={!ready || saving} onClick={save}>
          {saving ? "Saving…" : "Capture & Save"}
        </button>
        {status && <p className="text-xs text-text-muted mt-2">{status}</p>}
      </div>
    </div>
  );
}
