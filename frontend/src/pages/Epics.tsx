import { useEffect, useState } from "react";
import { api } from "../api";

export default function Epics() {
  const [projects, setProjects] = useState<any[]>([]);
  const [projectId, setProjectId] = useState<number | "">("");
  const [epics, setEpics] = useState<any[]>([]);
  const [progressByEpic, setProgressByEpic] = useState<Record<number, any>>({});
  const [newTitle, setNewTitle] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    api.projects().then((p) => { setProjects(p); if (p[0]) setProjectId(p[0].id); });
  }, []);

  useEffect(() => {
    if (!projectId) return;
    api.epics(projectId).then(async (list) => {
      setEpics(list);
      const entries = await Promise.all(list.map((e: any) => api.epicProgress(e.id).then((p: any) => [e.id, p])));
      setProgressByEpic(Object.fromEntries(entries));
    });
  }, [projectId]);

  const createEpic = async () => {
    if (!newTitle.trim() || !projectId) return;
    setCreating(true);
    try {
      await api.createEpic({ title: newTitle, project_id: projectId });
      setNewTitle("");
      const list = await api.epics(projectId);
      setEpics(list);
      const entries = await Promise.all(list.map((e: any) => api.epicProgress(e.id).then((p: any) => [e.id, p])));
      setProgressByEpic(Object.fromEntries(entries));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-semibold">Epics</h1>
          <p className="text-text-muted text-sm mt-1">Group related stories, tasks, and bugs under a shared initiative.</p>
        </div>
        <select className="border border-line rounded-lg px-3 py-2 text-sm" value={projectId} onChange={(e) => setProjectId(Number(e.target.value))}>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </div>

      <div className="card p-4 mb-6 flex gap-2">
        <input
          className="flex-1 border border-line rounded-lg px-3 py-2 text-sm outline-none focus:border-brand"
          placeholder="New epic title…"
          value={newTitle}
          onChange={(e) => setNewTitle(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && createEpic()}
        />
        <button className="btn-primary" disabled={creating || !newTitle.trim()} onClick={createEpic}>Create Epic</button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {epics.map((epic) => {
          const p = progressByEpic[epic.id];
          return (
            <div key={epic.id} className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: epic.color }} />
                <span className="text-xs text-text-muted font-mono">{epic.key}</span>
              </div>
              <div className="font-medium text-sm mb-3">{epic.title}</div>
              {p && (
                <>
                  <div className="h-2 bg-paper rounded-full overflow-hidden mb-2">
                    <div className="h-full bg-brand rounded-full" style={{ width: `${p.progress_pct}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-text-muted">
                    <span>{p.issue_count} issues</span>
                    <span>{p.completed_points}/{p.total_points} pts · {p.progress_pct}%</span>
                  </div>
                </>
              )}
            </div>
          );
        })}
        {epics.length === 0 && <div className="text-sm text-text-muted">No epics yet for this project.</div>}
      </div>
    </div>
  );
}
