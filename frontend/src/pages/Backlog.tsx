import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { PriorityPill } from "../components/Bits";

export default function Backlog() {
  const navigate = useNavigate();
  const [issues, setIssues] = useState<any[]>([]);
  const [sprints, setSprints] = useState<any[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [projectId, setProjectId] = useState<number | "">("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.projects().then((p) => { setProjects(p); if (p[0]) setProjectId(p[0].id); });
  }, []);

  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    Promise.all([api.issues({ project_id: projectId }), api.sprints(projectId)])
      .then(([i, s]) => {
        setIssues(i.filter((x: any) => !x.sprint_id && x.status !== "Done"));
        setSprints(s);
      })
      .finally(() => setLoading(false));
  }, [projectId]);

  const assignToSprint = async (issueId: number, sprintId: number) => {
    await api.updateIssue(issueId, { sprint_id: sprintId });
    setIssues((prev) => prev.filter((i) => i.id !== issueId));
  };

  const totalPoints = issues.reduce((sum, i) => sum + i.story_points, 0);

  return (
    <div className="p-8 max-w-4xl">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-semibold">Backlog</h1>
          <p className="text-text-muted text-sm mt-1">Unassigned work, sorted by priority — assign directly into a sprint.</p>
        </div>
        <select className="border border-line rounded-lg px-3 py-2 text-sm" value={projectId} onChange={(e) => setProjectId(Number(e.target.value))}>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </div>

      {loading ? (
        <div className="text-sm text-text-muted">Loading backlog…</div>
      ) : issues.length === 0 ? (
        <div className="card p-6 text-sm text-text-muted">No unassigned backlog items for this project — everything is either in a sprint or done.</div>
      ) : (
        <>
          <div className="text-xs text-text-muted mb-3">{issues.length} item(s) · {totalPoints} story points</div>
          <div className="card divide-y divide-line">
            {issues
              .sort((a, b) => (PRIORITY_RANK[b.priority] ?? 0) - (PRIORITY_RANK[a.priority] ?? 0))
              .map((issue) => (
                <div key={issue.id} className="p-4 flex items-center justify-between gap-4">
                  <div className="min-w-0">
                    <div className="text-sm font-medium truncate hover:text-brand cursor-pointer" onClick={() => navigate(`/issues/${issue.id}`)}>
                      {issue.key && <span className="text-xs font-mono text-text-muted mr-1">{issue.key}</span>}
                      {issue.title}
                    </div>
                    <div className="text-xs text-text-muted mt-0.5">{issue.issue_type} · {issue.story_points} pts</div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <PriorityPill value={issue.priority} />
                    <select
                      className="text-xs border border-line rounded-lg px-2 py-1.5"
                      defaultValue=""
                      onChange={(e) => e.target.value && assignToSprint(issue.id, Number(e.target.value))}
                    >
                      <option value="" disabled>Add to sprint…</option>
                      {sprints.filter((s) => s.status !== "completed").map((s) => (
                        <option key={s.id} value={s.id}>{s.name} ({s.status})</option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  );
}

const PRIORITY_RANK: Record<string, number> = { Critical: 4, High: 3, Medium: 2, Low: 1 };
