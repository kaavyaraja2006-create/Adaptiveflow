export function AIBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="badge-ai">
      <span className="w-1.5 h-1.5 rounded-full bg-brand" />
      {children}
    </span>
  );
}

const PRIORITY_COLORS: Record<string, string> = {
  HIGH: "bg-danger/10 text-danger", High: "bg-danger/10 text-danger", Critical: "bg-danger/10 text-danger",
  MEDIUM: "bg-warning/10 text-warning", Medium: "bg-warning/10 text-warning",
  LOW: "bg-text-muted/10 text-text-muted", Low: "bg-text-muted/10 text-text-muted",
};

export function PriorityPill({ value }: { value: string }) {
  return <span className={`badge ${PRIORITY_COLORS[value] || "bg-text-muted/10 text-text-muted"}`}>{value}</span>;
}

const STATUS_COLORS: Record<string, string> = {
  "To Do": "bg-text-muted/10 text-text-muted",
  "In Progress": "bg-brand-dim text-brand-deep",
  "In Review": "bg-warning/10 text-warning",
  "Done": "bg-success/10 text-success",
};

export function StatusPill({ value }: { value: string }) {
  return <span className={`badge ${STATUS_COLORS[value] || "bg-text-muted/10 text-text-muted"}`}>{value}</span>;
}

export function Avatar({ name, color }: { name: string; color?: string }) {
  const initials = name.split(" ").map((p) => p[0]).slice(0, 2).join("");
  return (
    <div
      className="w-7 h-7 rounded-full flex items-center justify-center text-white text-xs font-semibold shrink-0"
      style={{ backgroundColor: color || "#4F5DFF" }}
      title={name}
    >
      {initials}
    </div>
  );
}

const RISK_BAND_COLORS: Record<string, string> = {
  low: "bg-success/10 text-success",
  moderate: "bg-warning/10 text-warning",
  elevated: "bg-warning/20 text-warning",
  severe: "bg-danger/10 text-danger",
  opted_out: "bg-text-muted/10 text-text-muted",
  insufficient_data: "bg-text-muted/10 text-text-muted",
};

const RISK_BAND_LABELS: Record<string, string> = {
  low: "Low",
  moderate: "Moderate",
  elevated: "Elevated",
  severe: "Severe",
  opted_out: "Opted out",
  insufficient_data: "No data yet",
};

/** Sustainable Workload / Recovery Risk badge — a work-pattern signal only,
 * never a medical or emotional assessment. See ai/workload_engine.py. */
export function RecoveryRiskBadge({ band }: { band: string }) {
  return (
    <span className={`badge ${RISK_BAND_COLORS[band] || "bg-text-muted/10 text-text-muted"}`} title="Sustainable Workload / Recovery Risk — a work-pattern signal, not a medical assessment">
      Recovery risk: {RISK_BAND_LABELS[band] || band}
    </span>
  );
}
