import { useEffect, useState } from "react";
import { api } from "../api";
import { useAuth } from "../auth";

const ROLE_DESCRIPTIONS: Record<string, string> = {
  Admin: "Full access to all projects, teams, and AI settings",
  "Project Manager": "Manages sprints, roadmaps, and reassignment approvals",
  Developer: "Views assigned work and personal capability profile",
  Viewer: "Read-only access to dashboards and reports",
};

export default function Login() {
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { setUser } = useAuth();

  useEffect(() => {
    api.authUsers().then(setUsers).finally(() => setLoading(false));
  }, []);

  const login = async (id: number) => {
    const u = await api.loginAs(id);
    setUser(u);
  };

  return (
    <div className="min-h-screen bg-paper flex items-center justify-center p-8">
      <div className="w-full max-w-md">
        <div className="flex items-center gap-2 justify-center mb-8">
          <div className="w-8 h-8 rounded-md bg-brand flex items-center justify-center font-display font-bold text-white">A</div>
          <span className="font-display font-semibold text-xl tracking-tight">AdaptiveFlow</span>
        </div>

        <div className="card p-6">
          <h1 className="font-display font-semibold text-lg mb-1">Choose a demo user</h1>
          <p className="text-text-muted text-sm mb-5">
            This is a demo login — no password, just picking who you're acting as. Not a real security boundary.
          </p>

          {loading ? (
            <div className="text-sm text-text-muted">Loading users…</div>
          ) : (
            <div className="space-y-2">
              {users.map((u) => (
                <button
                  key={u.id}
                  onClick={() => login(u.id)}
                  className="w-full text-left border border-line rounded-lg px-4 py-3 hover:border-brand hover:bg-brand-dim/40 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">{u.name}</span>
                    <span className="badge bg-paper text-text-muted">{u.role}</span>
                  </div>
                  <div className="text-xs text-text-muted mt-0.5">{ROLE_DESCRIPTIONS[u.role] || u.email}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
