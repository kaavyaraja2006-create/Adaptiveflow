from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- Teams / Employees ----------

class TeamOut(ORMModel):
    id: int
    name: str
    description: str


class TeamIn(BaseModel):
    name: str
    description: str = ""


class WorkHistoryOut(ORMModel):
    id: int
    task_type: str
    story_points: int
    success: bool
    on_time: bool
    completed_at: datetime


class EmployeeOut(ORMModel):
    id: int
    name: str
    role: str
    avatar_color: str
    team_id: Optional[int]
    status: str
    utilization: float
    skills: Dict[str, Any]
    office_id: Optional[str] = None
    face_enrolled: bool = False
    workload_opt_out: bool = False


class EmployeeIn(BaseModel):
    name: str
    role: str = "Engineer"
    team_id: Optional[int] = None
    skills: Dict[str, int] = {}
    office_id: Optional[str] = None


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    team_id: Optional[int] = None
    skills: Optional[Dict[str, int]] = None
    utilization: Optional[float] = None
    status: Optional[str] = None
    workload_opt_out: Optional[bool] = None


class FaceEnrollIn(BaseModel):
    descriptor: List[float]


class BiometricCheckInIn(BaseModel):
    descriptor: List[float]


class ManualCheckInIn(BaseModel):
    employee_id: int


class ManualBulkCheckInIn(BaseModel):
    employee_ids: List[int]


class SubmitCheckInIn(BaseModel):
    present_employee_ids: List[int]


class AttendanceOut(ORMModel):
    id: int
    employee_id: int
    method: str
    match_distance: Optional[float]
    checked_in_at: datetime


class CheckInResult(BaseModel):
    matched: bool
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    match_distance: Optional[float] = None
    message: str


# ---------- Projects / Sprints ----------

class ProjectOut(ORMModel):
    id: int
    name: str
    description: str
    goal: str


class SprintOut(ORMModel):
    id: int
    name: str
    project_id: int
    status: str
    planned_points: int
    completed_points: int
    carry_over_points: int
    unplanned_points: int


# ---------- Ideas ----------

class IdeaIn(BaseModel):
    title: str
    description: str = ""
    project_id: Optional[int] = None


class IdeaUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    team: Optional[str] = None
    goal: Optional[str] = None


class IdeaOut(ORMModel):
    id: int
    title: str
    description: str
    problem: str
    project_id: Optional[int]
    status: str
    team: str
    goal: str
    impact: int
    effort: int
    confidence: int
    priority: str
    priority_reason: str
    created_at: datetime


class IdeaAnalysisOut(BaseModel):
    problem: str
    feature: str
    user: str
    category: str
    urgency: str
    goal: str
    team: str
    impact: int
    effort: int
    confidence: int
    priority: str
    priority_reason: str
    similar: List[Dict[str, Any]]
    duplicate: Optional[Dict[str, Any]]


# ---------- Issues ----------

class IssueIn(BaseModel):
    title: str
    description: str = ""
    issue_type: str = "Backend"
    item_type: str = "Task"
    project_id: Optional[int] = None
    story_points: int = 3
    priority: str = "Medium"
    epic_id: Optional[int] = None
    parent_id: Optional[int] = None


class IssueUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee_id: Optional[int] = None
    sprint_id: Optional[int] = None
    story_points: Optional[int] = None
    blocked: Optional[bool] = None
    epic_id: Optional[int] = None
    item_type: Optional[str] = None
    label_ids: Optional[List[int]] = None
    component_ids: Optional[List[int]] = None


class LabelOut(ORMModel):
    id: int
    name: str
    color: str


class ComponentOut(ORMModel):
    id: int
    name: str
    project_id: Optional[int]


class IssueOut(ORMModel):
    id: int
    key: Optional[str]
    title: str
    description: str
    issue_type: str
    item_type: str
    status: str
    priority: str
    story_points: int
    project_id: Optional[int]
    sprint_id: Optional[int]
    assignee_id: Optional[int]
    epic_id: Optional[int]
    parent_id: Optional[int]
    blocked: bool
    created_at: datetime
    labels: List[LabelOut] = []
    components: List[ComponentOut] = []


class EpicIn(BaseModel):
    title: str
    description: str = ""
    project_id: int
    color: str = "#4F5DFF"


class EpicOut(ORMModel):
    id: int
    key: Optional[str]
    title: str
    description: str
    project_id: int
    status: str
    color: str
    created_at: datetime


class EpicProgress(BaseModel):
    epic: EpicOut
    total_points: int
    completed_points: int
    remaining_points: int
    issue_count: int
    progress_pct: int


class CommentIn(BaseModel):
    author_name: str = "You"
    body: str


class CommentOut(ORMModel):
    id: int
    issue_id: int
    author_name: str
    body: str
    created_at: datetime
    edited_at: Optional[datetime]


class WatcherIn(BaseModel):
    employee_id: int


class WatcherOut(ORMModel):
    id: int
    issue_id: int
    employee_id: int
    employee_name: Optional[str] = None


class IssueLinkIn(BaseModel):
    target_issue_id: int
    link_type: str = "relates to"


class IssueLinkOut(ORMModel):
    id: int
    source_issue_id: int
    target_issue_id: int
    link_type: str
    target_title: Optional[str] = None
    target_key: Optional[str] = None


class ActivityLogOut(ORMModel):
    id: int
    issue_id: int
    actor: str
    field: str
    old_value: Optional[str]
    new_value: Optional[str]
    created_at: datetime


class LabelIn(BaseModel):
    name: str
    color: str = "#6B7080"


class ComponentIn(BaseModel):
    name: str
    project_id: int


# ---------- Absence / Assignment ----------

class AbsenceRequest(BaseModel):
    employee_id: int


class ReallocationAnalyzeRequest(BaseModel):
    task_id: int
    emergency_mode: bool = False


class ModelExplanation(BaseModel):
    top_positive_factors: List[str] = []
    top_negative_factors: List[str] = []


class CandidateScore(BaseModel):
    employee_id: int
    name: str
    skill: int
    success_rate: int
    capacity: int
    completion_probability: int
    fairness_penalty: int
    overall: int
    history_count: int
    # ---- extended (Phase 1-2: human-aware allocation upgrade) ----
    workload_ratio: Optional[float] = None
    recovery_risk: Optional[float] = None
    recovery_risk_band: Optional[str] = None
    workload_model_status: Optional[str] = None
    baseline_recovery_risk: Optional[float] = None
    model_recovery_risk: Optional[float] = None
    recovery_risk_gate: Optional[dict] = None
    allocation_score: Optional[int] = None
    score_version: Optional[str] = None
    score_breakdown: Optional[dict] = None
    recent_reassignments: Optional[int] = None
    transfer_cost: Optional[float] = None
    excluded_by_rule: bool = False
    exclusion_reason: Optional[str] = None
    requires_manager_review: bool = False
    review_reasons: List[str] = []
    applied_rule_ids: List[str] = []
    policy_explanation: List[str] = []
    model_explanation: Optional[ModelExplanation] = None


class ReassignmentResult(BaseModel):
    task_id: int
    task_title: str
    no_suitable_candidate: bool
    recommended_employee_id: Optional[int] = None
    recommended_employee_name: Optional[str] = None
    confidence: Optional[int] = None
    reason: Optional[str] = None
    alternatives: List[str] = []
    candidates: List[CandidateScore] = []
    # ---- extended ----
    confidence_label: Optional[str] = None
    data_quality_flags: List[str] = []
    rule_excluded_count: int = 0
    model_explanation: Optional[ModelExplanation] = None
    policy_explanation: List[str] = []
    requires_manager_review: bool = False
    review_reasons: List[str] = []
    transfer_cost: Optional[float] = None


# ---------- Workload / Recovery Risk / Rules (Phase 1-2) ----------

class WorkloadProfileOut(BaseModel):
    employee_id: int
    name: str
    opted_out: bool
    model_status: str
    model_version: str
    load_ratio: float
    context_switching: float
    high_complexity_density: float
    after_hours_ratio: float
    rest_days_recent: int
    recovery_risk: float
    risk_band: str
    expected_velocity_next_7_days: float
    prediction_confidence: float
    days_of_data: int
    stale_data_flags: List[str] = []
    baseline_recovery_risk: Optional[float] = None
    model_recovery_risk: Optional[float] = None
    gate: Optional[dict] = None
    current_day_risk: Optional[float] = None


class AllocationRuleOut(ORMModel):
    id: int
    code: str
    name: str
    description: str
    kind: str
    enabled: bool
    threshold: Optional[float] = None
    weight: Optional[float] = None


class AllocationRuleUpdate(BaseModel):
    enabled: Optional[bool] = None
    threshold: Optional[float] = None
    weight: Optional[float] = None


# ---------- Recommendations / Decisions ----------

class RecommendationOut(ORMModel):
    id: int
    kind: str
    title: str
    reason: str
    evidence: str
    confidence: int
    status: str
    related_issue_id: Optional[int]
    related_idea_id: Optional[int]
    created_at: datetime


class RecommendationDecision(BaseModel):
    decision: str  # approved | rejected | modified


class DecisionHistoryOut(ORMModel):
    id: int
    recommendation: str
    model: str
    confidence: int
    evidence: str
    human_decision: str
    outcome: str
    predicted_value: str
    actual_value: str
    created_at: datetime


# ---------- Dashboard / Risk / Simulation ----------

class ProjectHealthOut(BaseModel):
    project_id: int
    project_name: str
    health_score: int
    schedule_health: int
    risk_health: int
    resource_health: int
    progress_pct: int
    delay_probability: int
    predicted_delay_days: int
    reasons: List[str]


class ScenarioRequest(BaseModel):
    scenario_type: str  # move_sprint | remove_feature | add_developer | absence_extended | change_methodology
    project_id: int
    params: Dict[str, Any] = {}


class CompareBox(BaseModel):
    label: str
    value: int
    verdict: str
    tone: str  # good | bad | neutral
    recommended: bool = False


class WhyItem(BaseModel):
    icon: str
    value: str
    label: str


class ScenarioRecommendation(BaseModel):
    label: str
    confidence: int


class ScenarioResult(BaseModel):
    scenario_type: str
    title: str
    headline: str
    compare: List[CompareBox]
    why: List[WhyItem]
    insight: str
    recommendation: ScenarioRecommendation
