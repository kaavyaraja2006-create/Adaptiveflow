import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api";
import { Avatar } from "../components/Bits";

export default function IssueDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const issueId = Number(id);

  const [issue, setIssue] = useState<any>(null);
  const [comments, setComments] = useState<any[]>([]);
  const [watchers, setWatchers] = useState<any[]>([]);
  const [links, setLinks] = useState<any[]>([]);
  const [activity, setActivity] = useState<any[]>([]);
  const [employees, setEmployees] = useState<any[]>([]);
  const [allLabels, setAllLabels] = useState<any[]>([]);
  const [newComment, setNewComment] = useState("");
  const [linkTargetId, setLinkTargetId] = useState("");
  const [linkType, setLinkType] = useState("relates to");

  const load = () => {
    api.getIssue(issueId).then(setIssue);
    api.comments(issueId).then(setComments);
    api.watchers(issueId).then(setWatchers);
    api.issueLinks(issueId).then(setLinks);
    api.activity(issueId).then(setActivity);
  };

  useEffect(() => {
    load();
    api.employees().then(setEmployees);
    api.labels().then(setAllLabels);
  }, [issueId]);

  if (!issue) return <div className="p-8 text-text-muted">Loading issue…</div>;

  const assignee = employees.find((e) => e.id === issue.assignee_id);

  const updateField = async (field: string, value: any) => {
    const updated = await api.updateIssue(issueId, { [field]: value });
    setIssue(updated);
    api.activity(issueId).then(setActivity);
  };

  const postComment = async () => {
    if (!newComment.trim()) return;
    await api.addComment(issueId, "You", newComment);
    setNewComment("");
    api.comments(issueId).then(setComments);
    api.activity(issueId).then(setActivity);
  };

  const toggleWatch = async (employeeId: number) => {
    const already = watchers.find((w: any) => w.employee_id === employeeId);
    if (already) await api.removeWatcher(issueId, employeeId);
    else await api.addWatcher(issueId, employeeId);
    api.watchers(issueId).then(setWatchers);
  };

  const addLink = async () => {
    if (!linkTargetId) return;
    await api.addIssueLink(issueId, Number(linkTargetId), linkType);
    setLinkTargetId("");
    api.issueLinks(issueId).then(setLinks);
  };

  const toggleLabel = async (labelId: number) => {
    const has = issue.labels.some((l: any) => l.id === labelId);
    const nextIds = has ? issue.labels.filter((l: any) => l.id !== labelId).map((l: any) => l.id) : [...issue.labels.map((l: any) => l.id), labelId];
    const updated = await api.updateIssue(issueId, { label_ids: nextIds });
    setIssue(updated);
  };

  return (
    <div className="p-8 max-w-4xl">
      <button className="text-xs text-text-muted underline mb-4" onClick={() => navigate(-1)}>← Back</button>

      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-mono text-text-muted">{issue.key}</span>
        <span className="badge bg-paper text-text-muted">{issue.item_type}</span>
      </div>
      <h1 className="font-display text-2xl font-semibold mb-4">{issue.title}</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="card p-4">
            <div className="text-xs text-text-muted mb-1">Description</div>
            <p className="text-sm">{issue.description || "No description."}</p>
          </div>

          <div className="card p-4">
            <div className="text-xs text-text-muted mb-2">Labels</div>
            <div className="flex flex-wrap gap-2">
              {allLabels.map((l) => {
                const active = issue.labels.some((il: any) => il.id === l.id);
                return (
                  <button
                    key={l.id}
                    onClick={() => toggleLabel(l.id)}
                    className={`badge ${active ? "text-white" : "bg-paper text-text-muted"}`}
                    style={active ? { backgroundColor: l.color } : {}}
                  >
                    {l.name}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs text-text-muted mb-2">Linked Issues</div>
            <div className="space-y-1.5 mb-3">
              {links.length === 0 && <div className="text-sm text-text-muted">No links yet.</div>}
              {links.map((l) => (
                <div key={l.id} className="text-sm flex items-center gap-2">
                  <span className="badge bg-paper text-text-muted">{l.link_type}</span>
                  <span className="font-mono text-xs text-text-muted">{l.target_key}</span>
                  {l.target_title}
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <select className="text-xs border border-line rounded-lg px-2 py-1.5" value={linkType} onChange={(e) => setLinkType(e.target.value)}>
                {["blocks", "blocked by", "relates to", "duplicates"].map((t) => <option key={t}>{t}</option>)}
              </select>
              <input
                className="flex-1 text-xs border border-line rounded-lg px-2 py-1.5"
                placeholder="Target issue ID…"
                value={linkTargetId}
                onChange={(e) => setLinkTargetId(e.target.value)}
              />
              <button className="btn-ghost !px-3 !py-1.5 text-xs" onClick={addLink}>Add Link</button>
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs text-text-muted mb-3">Comments ({comments.length})</div>
            <div className="space-y-3 mb-3">
              {comments.map((c) => (
                <div key={c.id} className="bg-paper rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{c.author_name}</span>
                    <span className="text-xs text-text-muted">{new Date(c.created_at).toLocaleString()}</span>
                  </div>
                  <p className="text-sm">{c.body}</p>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                className="flex-1 border border-line rounded-lg px-3 py-2 text-sm"
                placeholder="Add a comment…"
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && postComment()}
              />
              <button className="btn-primary" onClick={postComment}>Post</button>
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs text-text-muted mb-3">Activity</div>
            <div className="space-y-2">
              {activity.length === 0 && <div className="text-sm text-text-muted">No activity yet.</div>}
              {activity.map((a) => (
                <div key={a.id} className="text-xs text-text-muted flex items-center gap-1.5">
                  <span className="font-medium text-text">{a.actor}</span>
                  {a.field === "comment" ? "commented" : a.field === "created" ? "created this issue" : `changed ${a.field} from ${a.old_value ?? "—"} to ${a.new_value ?? "—"}`}
                  <span>· {new Date(a.created_at).toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-4 space-y-3">
            <div>
              <div className="text-xs text-text-muted mb-1">Status</div>
              <select className="w-full border border-line rounded-lg px-2 py-1.5 text-sm" value={issue.status} onChange={(e) => updateField("status", e.target.value)}>
                {["To Do", "In Progress", "In Review", "Done"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <div className="text-xs text-text-muted mb-1">Priority</div>
              <select className="w-full border border-line rounded-lg px-2 py-1.5 text-sm" value={issue.priority} onChange={(e) => updateField("priority", e.target.value)}>
                {["Low", "Medium", "High", "Critical"].map((p) => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <div className="text-xs text-text-muted mb-1">Assignee</div>
              <select className="w-full border border-line rounded-lg px-2 py-1.5 text-sm" value={issue.assignee_id || ""} onChange={(e) => updateField("assignee_id", Number(e.target.value))}>
                <option value="">Unassigned</option>
                {employees.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
              </select>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-line text-xs">
              <span className="text-text-muted">Story points</span>
              <span className="font-mono">{issue.story_points}</span>
            </div>
          </div>

          <div className="card p-4">
            <div className="text-xs text-text-muted mb-2">Watchers ({watchers.length})</div>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {watchers.map((w: any) => <Avatar key={w.id} name={w.employee_name} />)}
            </div>
            <select
              className="w-full text-xs border border-line rounded-lg px-2 py-1.5"
              value=""
              onChange={(e) => e.target.value && toggleWatch(Number(e.target.value))}
            >
              <option value="" disabled>Add/remove watcher…</option>
              {employees.map((e) => (
                <option key={e.id} value={e.id}>
                  {watchers.some((w: any) => w.employee_id === e.id) ? "✓ " : ""}{e.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
