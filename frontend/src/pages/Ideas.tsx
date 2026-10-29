import { useEffect, useState } from "react";
import { api } from "../api";
import ConfidenceRing from "../components/ConfidenceRing";
import { AIBadge, PriorityPill } from "../components/Bits";

export default function Ideas() {
  const [ideas, setIdeas] = useState<any[]>([]);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [analysis, setAnalysis] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const load = () => api.ideas().then(setIdeas);
  useEffect(() => { load(); }, []);

  const runAnalysis = async () => {
    if (!title.trim()) return;
    setAnalyzing(true);
    try {
      const result = await api.analyzeIdea(title, description);
      setAnalysis(result);
    } finally {
      setAnalyzing(false);
    }
  };

  const accept = async () => {
    await api.createIdea(title, description);
    setTitle(""); setDescription(""); setAnalysis(null);
    load();
  };

  return (
    <div className="p-8 max-w-6xl space-y-8">
      <div>
        <h1 className="font-display text-2xl font-semibold">Ideas</h1>
        <p className="text-text-muted text-sm mt-1">Describe an idea in plain language — AI extracts structure, predicts impact and effort, and checks for duplicates.</p>
      </div>

      <div className="card p-5 space-y-3">
        <input
          className="w-full border border-line rounded-lg px-3 py-2 text-sm outline-none focus:border-brand"
          placeholder="e.g. Customers are unable to reschedule appointments"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <textarea
          className="w-full border border-line rounded-lg px-3 py-2 text-sm outline-none focus:border-brand"
          placeholder="Add more detail (optional)"
          rows={2}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <button className="btn-primary" disabled={!title.trim() || analyzing} onClick={runAnalysis}>
          {analyzing ? "Analyzing…" : "Analyze with AI"}
        </button>

        {analysis && (
          <div className="border border-brand/30 bg-brand-dim/40 rounded-xl2 p-4 mt-2 space-y-3">
            <AIBadge>AI Interpretation</AIBadge>

            {analysis.duplicate && (
              <div className="bg-warning/10 text-warning text-sm rounded-lg px-3 py-2">
                Potential duplicate: “{analysis.duplicate.title}” — {analysis.duplicate.pct}% similar.
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div><div className="text-xs text-text-muted">Category</div>{analysis.category}</div>
              <div><div className="text-xs text-text-muted">Team</div>{analysis.team}</div>
              <div><div className="text-xs text-text-muted">Goal</div>{analysis.goal}</div>
              <div><div className="text-xs text-text-muted">Urgency</div>{analysis.urgency}</div>
            </div>

            <div className="flex items-center gap-6 flex-wrap">
              <div className="flex items-center gap-2">
                <ConfidenceRing value={analysis.impact} size={40} />
                <span className="text-xs text-text-muted">Impact</span>
              </div>
              <div className="text-sm">
                Effort: <span className="font-mono font-semibold">{analysis.effort} pts</span>
              </div>
              <div className="text-sm flex items-center gap-2">
                Priority: <PriorityPill value={analysis.priority} />
              </div>
              <div className="flex items-center gap-2">
                <ConfidenceRing value={analysis.confidence} size={36} />
                <span className="text-xs text-text-muted">Confidence</span>
              </div>
            </div>
            <p className="text-xs text-text-muted">{analysis.priority_reason}</p>

            {analysis.similar?.length > 0 && (
              <div>
                <div className="text-xs text-text-muted mb-1">Similar ideas</div>
                <div className="flex flex-wrap gap-2">
                  {analysis.similar.map((s: any) => (
                    <span key={s.idea_id} className="badge bg-white border border-line">{s.title} — {s.pct}%</span>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <button className="btn-primary" onClick={accept}>Accept</button>
              <button className="btn-ghost" onClick={() => setAnalysis(null)}>Reject</button>
            </div>
          </div>
        )}
      </div>

      <div className="space-y-2">
        <h2 className="font-display font-semibold text-sm text-text-muted uppercase tracking-wide">All Ideas ({ideas.length})</h2>
        <div className="card divide-y divide-line">
          {ideas.map((idea) => (
            <div key={idea.id} className="p-4 flex items-center justify-between gap-4">
              <div className="min-w-0">
                <div className="text-sm font-medium truncate">{idea.title}</div>
                <div className="text-xs text-text-muted mt-0.5">{idea.team} · {idea.goal}</div>
              </div>
              <div className="flex items-center gap-4 shrink-0">
                <PriorityPill value={idea.priority} />
                <span className="text-xs font-mono text-text-muted">Impact {idea.impact}</span>
                <ConfidenceRing value={idea.confidence} size={32} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
