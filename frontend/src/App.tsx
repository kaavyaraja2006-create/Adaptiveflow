import { HashRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Ideas from "./pages/Ideas";
import Board from "./pages/Board";
import Backlog from "./pages/Backlog";
import Epics from "./pages/Epics";
import IssueDetail from "./pages/IssueDetail";
import Team from "./pages/Team";
import CheckIn from "./pages/CheckIn";
import AbsenceRecovery from "./pages/AbsenceRecovery";
import Simulator from "./pages/Simulator";
import DecisionHistory from "./pages/DecisionHistory";
import AuditVerification from "./pages/AuditVerification";
import { AuthProvider, useAuth } from "./auth";

function AppShell() {
  const { user } = useAuth();

  if (!user) return <Login />;

  return (
    <HashRouter>
      <div className="flex min-h-screen bg-paper">
        <Sidebar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/ideas" element={<Ideas />} />
            <Route path="/board" element={<Board />} />
            <Route path="/backlog" element={<Backlog />} />
            <Route path="/epics" element={<Epics />} />
            <Route path="/issues/:id" element={<IssueDetail />} />
            <Route path="/team" element={<Team />} />
            <Route path="/checkin" element={<CheckIn />} />
            <Route path="/absence" element={<AbsenceRecovery />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/decisions" element={<DecisionHistory />} />
            <Route path="/audit" element={<AuditVerification />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
