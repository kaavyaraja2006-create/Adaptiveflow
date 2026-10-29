import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ensureModelsLoaded, startWebcam, stopWebcam, captureDescriptor } from "../face";
import { AIBadge, Avatar } from "../components/Bits";
import ConfidenceRing from "../components/ConfidenceRing";

export default function CheckIn() {
  const [mode, setMode] = useState<"biometric" | "manual">("biometric");
  const [today, setToday] = useState<any[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [status, setStatus] = useState<string>("");
  const [modelsReady, setModelsReady] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [decided, setDecided] = useState<Record<number, string>>({});

  const decide = async (r: any, approve: boolean) => {
    if (approve) await api.approveReassignment(r.task_id, r.recommended_employee_id);
    else await api.rejectReassignment(r.task_id, "Rejected at check-in review");
    setDecided((d) => ({ ...d, [r.task_id]: approve ? "Approved" : "Rejected" }));
  };

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const loadToday = () => api.attendanceToday().then((data) => {
    setToday(data);
    setSelectedIds((prev) => new Set([...prev].filter((id) => data.find((t: any) => t.employee_id === id && !t.checked_in))));
  });

  useEffect(() => { loadToday(); }, []);

  useEffect(() => {
    if (mode !== "biometric") {
      stopWebcam(streamRef.current);
      streamRef.current = null;
      return;
    }
    let cancelled = false;
    (async () => {
      setStatus("Loading face recognition models…");
      await ensureModelsLoaded();
      if (cancelled) return;
      setModelsReady(true);
      setStatus("");
      if (videoRef.current) {
        streamRef.current = await startWebcam(videoRef.current);
      }
    })();
    return () => {
      cancelled = true;
      stopWebcam(streamRef.current);
      streamRef.current = null;
    };
  }, [mode]);

  const captureAndCheckIn = async () => {
    if (!videoRef.current) return;
    setScanning(true);
    setStatus("Scanning face…");
    try {
      const descriptor = await captureDescriptor(videoRef.current);
      if (!descriptor) {
        setStatus("No face detected — make sure your face is centered and well lit, then try again.");
        return;
      }
      const res = await api.checkInBiometric(descriptor);
      setStatus(res.message);
      loadToday();
    } finally {
      setScanning(false);
    }
  };

  const notCheckedIn = today.filter((t) => !t.checked_in);

  const toggle = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const selectAll = () => setSelectedIds(new Set(notCheckedIn.map((t) => t.employee_id)));
  const clearAll = () => setSelectedIds(new Set());

  const unselectedCount = notCheckedIn.filter((t) => !selectedIds.has(t.employee_id) && t.status !== "absent").length;

  const submit = async () => {
    setSubmitting(true);
    setResult(null);
    setStatus(`Processing attendance for ${notCheckedIn.length} remaining employee(s)…`);
    try {
      const res = await api.submitCheckIn([...selectedIds]);
      setResult(res);
      setStatus(
        `Checked in ${res.checked_in.length}. ${res.auto_marked_absent.length} marked absent; their tasks have reassignment recommendations awaiting your approval.`
      );
      clearAll();
      loadToday();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold">Check-In</h1>
        <p className="text-text-muted text-sm mt-1">
          Whoever isn't checked in when you submit is marked absent, and the system recommends who should
          take over each of their tasks. Nothing is reassigned until you approve it.
        </p>
      </div>

      <div className="card p-5">
        <div className="flex gap-2 mb-4">
          <button className={`btn-ghost !px-3 !py-1.5 text-xs ${mode === "biometric" ? "!bg-brand !text-white !border-brand" : ""}`} onClick={() => setMode("biometric")}>Biometric</button>
          <button className={`btn-ghost !px-3 !py-1.5 text-xs ${mode === "manual" ? "!bg-brand !text-white !border-brand" : ""}`} onClick={() => setMode("manual")}>Manual</button>
        </div>

        {mode === "biometric" ? (
          <div className="space-y-3">
            <video ref={videoRef} className="rounded-lg bg-ink w-full max-w-xs" muted playsInline />
            <button className="btn-primary" disabled={!modelsReady || scanning} onClick={captureAndCheckIn}>
              {scanning ? "Scanning…" : "Capture & Check In"}
            </button>
            <p className="text-xs text-text-muted">Face data is stored as a numeric descriptor only — no photos or video are saved.</p>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex gap-2">
                <button className="text-xs underline text-brand" onClick={selectAll}>Select all present</button>
                <button className="text-xs underline text-text-muted" onClick={clearAll}>Clear</button>
              </div>
              <span className="text-xs text-text-muted">{selectedIds.size} marked present</span>
            </div>

            <div className="border border-line rounded-lg divide-y divide-line max-h-72 overflow-y-auto mb-3">
              {notCheckedIn.length === 0 && <div className="p-3 text-sm text-text-muted">Everyone has checked in.</div>}
              {notCheckedIn.map((t) => (
                <label key={t.employee_id} className="flex items-center justify-between gap-3 px-3 py-2 text-sm cursor-pointer hover:bg-paper">
                  <span className="flex items-center gap-2.5">
                    <input type="checkbox" checked={selectedIds.has(t.employee_id)} onChange={() => toggle(t.employee_id)} />
                    {t.name}
                  </span>
                  {t.status === "absent" && <AIBadge>Already processed as absent</AIBadge>}
                </label>
              ))}
            </div>

            <button className="btn-primary w-full" disabled={submitting || notCheckedIn.length === 0} onClick={submit}>
              {submitting ? "Processing…" : `Submit Check-In (${selectedIds.size} present, ${unselectedCount} will be marked absent)`}
            </button>
          </div>
        )}

        {status && <div className="text-sm bg-paper rounded-lg px-3 py-2 mt-3">{status}</div>}
      </div>

      {result && result.reassignments?.length > 0 && (
        <div className="space-y-3">
          <h2 className="font-display font-semibold text-sm text-text-muted uppercase tracking-wide">
            Reassignment recommendations ({result.reassignments.length})
          </h2>
          {result.reassignments.map((r: any, i: number) => (
            <div key={i} className="card p-4">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <div className="text-xs text-text-muted">{r.employee_absent} was absent — awaiting manager decision</div>
                  <div className="font-medium text-sm">{r.task_title}</div>
                </div>
                {r.no_suitable_candidate ? (
                  <AIBadge>Needs review</AIBadge>
                ) : (
                  <AIBadge>{decided[r.task_id] ?? "Pending approval"}</AIBadge>
                )}
              </div>

              {r.no_suitable_candidate ? (
                <div className="bg-warning/10 text-warning text-sm rounded-lg p-3">
                  {r.reason}
                  <div className="text-xs mt-1">No candidate met the viability threshold, so nothing was recommended.</div>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <Avatar name={r.recommended_employee_name} />
                  <div className="flex-1 text-xs text-text-muted">{r.reason}</div>
                  <ConfidenceRing value={r.confidence} size={36} />
                  {!decided[r.task_id] && (
                    <div className="flex gap-2">
                      <button className="btn-primary text-xs" onClick={() => decide(r, true)}>Approve</button>
                      <button className="btn-ghost text-xs" onClick={() => decide(r, false)}>Reject</button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
