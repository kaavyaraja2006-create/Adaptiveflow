const BASE = "http://localhost:8000/api";

async function req(path: string, options: RequestInit = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

export const api = {
  // teams / employees
  teams: () => req("/teams"),
  employees: (teamId?: number) => req(`/employees${teamId ? `?team_id=${teamId}` : ""}`),
  employeeProfile: (id: number) => req(`/employees/${id}/profile`),
  workload: (teamId?: number) => req(`/workload${teamId ? `?team_id=${teamId}` : ""}`),
  projects: () => req("/projects"),
  sprints: (projectId?: number) => req(`/sprints${projectId ? `?project_id=${projectId}` : ""}`),

  // ideas
  ideas: (projectId?: number) => req(`/ideas${projectId ? `?project_id=${projectId}` : ""}`),
  analyzeIdea: (title: string, description: string, project_id?: number) =>
    req("/ideas/analyze", { method: "POST", body: JSON.stringify({ title, description, project_id }) }),
  createIdea: (title: string, description: string, project_id?: number) =>
    req("/ideas", { method: "POST", body: JSON.stringify({ title, description, project_id }) }),

  // issues
  issues: (params: Record<string, any> = {}) => {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined) as any).toString();
    return req(`/issues${qs ? `?${qs}` : ""}`);
  },
  createIssue: (payload: any) => req("/issues", { method: "POST", body: JSON.stringify(payload) }),
  updateIssue: (id: number, payload: any) => req(`/issues/${id}`, { method: "PATCH", body: JSON.stringify(payload) }),
  getIssue: (id: number) => req(`/issues/${id}`),
  deleteIssue: (id: number) => req(`/issues/${id}`, { method: "DELETE" }),

  // epics
  epics: (projectId?: number) => req(`/epics${projectId ? `?project_id=${projectId}` : ""}`),
  createEpic: (payload: any) => req("/epics", { method: "POST", body: JSON.stringify(payload) }),
  epicProgress: (epicId: number) => req(`/epics/${epicId}/progress`),

  // comments
  comments: (issueId: number) => req(`/issues/${issueId}/comments`),
  addComment: (issueId: number, author_name: string, body: string) =>
    req(`/issues/${issueId}/comments`, { method: "POST", body: JSON.stringify({ author_name, body }) }),

  // watchers
  watchers: (issueId: number) => req(`/issues/${issueId}/watchers`),
  addWatcher: (issueId: number, employee_id: number) =>
    req(`/issues/${issueId}/watchers`, { method: "POST", body: JSON.stringify({ employee_id }) }),
  removeWatcher: (issueId: number, employeeId: number) =>
    req(`/issues/${issueId}/watchers/${employeeId}`, { method: "DELETE" }),

  // links
  issueLinks: (issueId: number) => req(`/issues/${issueId}/links`),
  addIssueLink: (issueId: number, target_issue_id: number, link_type: string) =>
    req(`/issues/${issueId}/links`, { method: "POST", body: JSON.stringify({ target_issue_id, link_type }) }),

  // activity
  activity: (issueId: number) => req(`/issues/${issueId}/activity`),

  // labels / components
  labels: () => req("/labels"),
  createLabel: (name: string) => req("/labels", { method: "POST", body: JSON.stringify({ name }) }),
  components: (projectId?: number) => req(`/components${projectId ? `?project_id=${projectId}` : ""}`),
  createComponent: (name: string, project_id: number) =>
    req("/components", { method: "POST", body: JSON.stringify({ name, project_id }) }),

  // absence
  markAbsent: (employee_id: number) => req("/absence/mark", { method: "POST", body: JSON.stringify({ employee_id }) }),
  returnFromAbsence: (employeeId: number) => req(`/absence/return/${employeeId}`, { method: "POST" }),
  approveReassignment: (issueId: number, employeeId: number) =>
    req(`/absence/reassign/${issueId}/${employeeId}`, { method: "POST" }),
  rejectReassignment: (issueId: number, reason = "") =>
    req(`/absence/reject/${issueId}?reason=${encodeURIComponent(reason)}`, { method: "POST" }),

  // human-aware allocation (Phase 1-2): workload / recovery risk / rules
  workloadProfile: (employeeId: number) => req(`/employees/${employeeId}/workload-profile`),
  recoveryRisk: (employeeId: number) => req(`/employees/${employeeId}/recovery-risk`),
  allocationRules: () => req("/allocation-rules"),
  updateAllocationRule: (code: string, payload: Record<string, any>) =>
    req(`/allocation-rules/${code}`, { method: "PATCH", body: JSON.stringify(payload) }),
  reallocationCandidates: (taskId: number) => req(`/ai/reallocation/${taskId}/candidates`),
  analyzeReallocation: (taskId: number, emergency_mode = false) =>
    req("/ai/reallocation/analyze", { method: "POST", body: JSON.stringify({ task_id: taskId, emergency_mode }) }),
  dataQualityScan: () => req("/data-quality/scan"),
  setWorkloadOptOut: (employeeId: number, workload_opt_out: boolean) =>
    req(`/employees/${employeeId}`, { method: "PATCH", body: JSON.stringify({ workload_opt_out }) }),
  sequenceModelStatus: () => req("/ai/models/status"),
  trainSequenceModel: () => req("/ai/models/train", { method: "POST" }),

  // digital twin (Phase 4-7)
  simulatorRun: (project_id: number, sprint_id?: number, iterations = 1000) =>
    req("/simulator/run", { method: "POST", body: JSON.stringify({ project_id, sprint_id, iterations }) }),
  simulatorWhatIf: (project_id: number, scenario: Record<string, any>, sprint_id?: number, iterations = 400) =>
    req("/simulator/whatif", { method: "POST", body: JSON.stringify({ project_id, sprint_id, scenario, iterations }) }),
  simulatorGraph: (projectId: number) => req(`/simulator/graph/${projectId}`),
  simulatorHistory: (projectId: number) => req(`/simulator/history/${projectId}`),
  durationModelStatus: () => req("/simulator/models/duration/status"),
  trainDurationModel: () => req("/simulator/models/duration/train", { method: "POST" }),
  simulateReallocationCandidates: (taskId: number) => req(`/ai/reallocation/${taskId}/simulate-candidates`),

  // audit ledger (Phase 8-13)
  taskAuditTrail: (taskId: number) => req(`/audit/tasks/${taskId}`),
  projectAuditTrail: (projectId: number) => req(`/audit/projects/${projectId}`),
  verifyAuditEvent: (eventId: string) => req(`/audit/verify/event/${eventId}`),
  verifyAuditBatch: (batchId: number) => req(`/audit/verify/batch/${batchId}`),
  verifyAuditChain: () => req("/audit/verify/chain"),
  anchorDailyBatch: () => req("/audit/anchor/daily", { method: "POST" }),
  listAuditBatches: () => req("/audit/batches"),
  auditPublicKey: () => req("/audit/public-key"),

  // recommendations / decisions
  recommendations: (status?: string) => req(`/recommendations${status ? `?status=${status}` : ""}`),
  decideRecommendation: (id: number, decision: string) =>
    req(`/recommendations/${id}/decide`, { method: "POST", body: JSON.stringify({ decision }) }),
  decisions: () => req("/decisions"),
  modelPerformance: () => req("/decisions/performance"),

  // dashboard / simulation
  dashboardHealth: () => req("/dashboard/health"),
  methodology: (projectId: number) => req(`/dashboard/methodology/${projectId}`),
  bottlenecks: (projectId: number) => req(`/dashboard/bottlenecks/${projectId}`),
  simulate: (scenario_type: string, project_id: number, params: Record<string, any>) =>
    req("/simulate", { method: "POST", body: JSON.stringify({ scenario_type, project_id, params }) }),

  // attendance / biometric
  enrollFace: (employeeId: number, descriptor: number[]) =>
    req(`/attendance/enroll/${employeeId}`, { method: "POST", body: JSON.stringify({ descriptor }) }),
  removeEnrollment: (employeeId: number) => req(`/attendance/enroll/${employeeId}`, { method: "DELETE" }),
  checkInBiometric: (descriptor: number[]) =>
    req("/attendance/check-in/biometric", { method: "POST", body: JSON.stringify({ descriptor }) }),
  checkInManual: (employee_id: number) =>
    req("/attendance/check-in/manual", { method: "POST", body: JSON.stringify({ employee_id }) }),
  checkInManualBulk: (employee_ids: number[]) =>
    req("/attendance/check-in/manual-bulk", { method: "POST", body: JSON.stringify({ employee_ids }) }),
  submitCheckIn: (present_employee_ids: number[]) =>
    req("/attendance/submit-checkin", { method: "POST", body: JSON.stringify({ present_employee_ids }) }),
  attendanceToday: () => req("/attendance/today"),

  // auth stub
  authUsers: () => req("/auth/users"),
  loginAs: (userId: number) => req(`/auth/login/${userId}`, { method: "POST" }),
};
