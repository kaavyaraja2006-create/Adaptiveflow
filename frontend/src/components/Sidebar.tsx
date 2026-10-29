import { NavLink } from "react-router-dom";
import { useEffect, useState } from "react";
import { useAuth } from "../auth";

const NAV = [
  { to: "/", label: "Dashboard", icon: "▦" },
  { to: "/ideas", label: "Ideas", icon: "✦" },
  { to: "/board", label: "Board", icon: "☷" },
  { to: "/backlog", label: "Backlog", icon: "☰" },
  { to: "/epics", label: "Epics", icon: "◈" },
  { to: "/team", label: "Team", icon: "◎" },
  { to: "/checkin", label: "Check-In", icon: "◉" },
  { to: "/absence", label: "Absence Recovery", icon: "✦", highlight: true },
  { to: "/simulator", label: "What-If Simulator", icon: "⚙" },
  { to: "/decisions", label: "Decision History", icon: "≡" },
  { to: "/audit", label: "Audit Verification", icon: "⛓" },
];

function useDarkMode() {
  const [dark, setDark] = useState(() => {
    const stored = localStorage.getItem("adaptiveflow-theme");
    if (stored) return stored === "dark";
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("adaptiveflow-theme", dark ? "dark" : "light");
  }, [dark]);

  return [dark, setDark] as const;
}

export default function Sidebar() {
  const [dark, setDark] = useDarkMode();
  const { user, setUser } = useAuth();

  return (
    <aside className="w-60 shrink-0 bg-ink text-white flex flex-col h-screen sticky top-0">
      <div className="px-5 py-5 flex items-center gap-2">
        <div className="w-7 h-7 rounded-md bg-brand flex items-center justify-center font-display font-bold text-sm">A</div>
        <span className="font-display font-semibold tracking-tight">AdaptiveFlow</span>
      </div>
      <nav className="flex-1 px-3 py-2 space-y-0.5">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? "bg-ink-light text-white"
                  : "text-white/60 hover:text-white hover:bg-ink-light/60"
              } ${item.highlight ? "relative" : ""}`
            }
          >
            <span className="w-4 text-center">{item.icon}</span>
            {item.label}
            {item.highlight && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-brand" />}
          </NavLink>
        ))}
      </nav>
      <button
        onClick={() => setDark(!dark)}
        className="mx-3 mb-2 flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-white/60 hover:text-white hover:bg-ink-light/60 transition-colors"
      >
        <span className="w-4 text-center">{dark ? "☀" : "☾"}</span>
        {dark ? "Light mode" : "Dark mode"}
      </button>
      <div className="px-5 py-4 border-t border-ink-border text-xs">
        <div className="flex items-center justify-between mb-1">
          <span className="text-white/70 font-medium">{user?.name}</span>
          <button className="text-white/40 underline hover:text-white/70" onClick={() => setUser(null)}>Switch</button>
        </div>
        <div className="text-white/40">{user?.role} · NovaCare Technologies</div>
      </div>
    </aside>
  );
}
