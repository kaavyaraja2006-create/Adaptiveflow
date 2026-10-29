import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { PriorityPill } from "../components/Bits";

const COLUMNS = ["To Do", "In Progress", "In Review", "Done"];

export default function Board() {
  const [issues, setIssues] = useState<any[]>([]);
  const [employees, setEmployees] = useState<any[]>([]);
  const [dragId, setDragId] = useState<number | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.issues().then(setIssues);
    api.employees().then(setEmployees);
  }, []);

  const empName = (id: number | null) => employees.find((e) => e.id === id)?.name || "Unassigned";

  const drop = async (status: string) => {
    if (dragId == null) return;
    setIssues((prev) => prev.map((i) => (i.id === dragId ? { ...i, status } : i)));
    await api.updateIssue(dragId, { status });
    setDragId(null);
  };

  return (
    <div className="p-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="font-display text-2xl font-semibold">Board</h1>
        <p className="text-text-muted text-sm mt-1">Drag cards between columns — changes persist to the database.</p>
      </div>
      <div className="grid grid-cols-4 gap-4">
        {COLUMNS.map((col) => {
          const colIssues = issues.filter((i) => i.status === col);
          return (
            <div
              key={col}
              className="bg-paper rounded-xl2 p-3 min-h-[60vh]"
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => drop(col)}
            >
              <div className="flex items-center justify-between mb-3 px-1">
                <span className="text-xs font-semibold text-text-muted uppercase tracking-wide">{col}</span>
                <span className="text-xs text-text-muted">{colIssues.length}</span>
              </div>
              <div className="space-y-2">
                {colIssues.map((issue) => (
                  <div
                    key={issue.id}
                    draggable
                    onDragStart={() => setDragId(issue.id)}
                    className="card p-3 cursor-grab active:cursor-grabbing"
                  >
                    <div className="text-sm font-medium mb-2 leading-snug hover:text-brand" onClick={(e) => { e.stopPropagation(); navigate(`/issues/${issue.id}`); }}>
                      {issue.key && <span className="text-xs font-mono text-text-muted mr-1">{issue.key}</span>}
                      {issue.title}
                    </div>
                    <div className="flex items-center justify-between">
                      <PriorityPill value={issue.priority} />
                      <span className="text-xs text-text-muted font-mono">{issue.story_points} pts</span>
                    </div>
                    <div className="text-xs text-text-muted mt-2 flex items-center justify-between">
                      <span>{empName(issue.assignee_id)}</span>
                      {issue.blocked && <span className="text-danger">Blocked</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
