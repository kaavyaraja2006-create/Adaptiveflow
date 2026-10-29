interface Props {
  value: number; // 0-100
  size?: number;
  label?: string;
}

export default function ConfidenceRing({ value, size = 44, label }: Props) {
  const stroke = 4;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (Math.min(100, Math.max(0, value)) / 100) * c;
  const color = value >= 75 ? "#1F9D6E" : value >= 50 ? "#4F5DFF" : "#C98A1B";

  return (
    <div className="flex items-center gap-2">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#E4E6EF" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        <text
          x="50%" y="50%" textAnchor="middle" dy="0.35em"
          className="font-mono font-semibold" fontSize={size * 0.28} fill="#1B1F30"
          transform={`rotate(90 ${size / 2} ${size / 2})`}
        >
          {Math.round(value)}
        </text>
      </svg>
      {label && <span className="text-xs text-text-muted">{label}</span>}
    </div>
  );
}
