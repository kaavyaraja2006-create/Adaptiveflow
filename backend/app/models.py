"""
Core relational schema for AdaptiveFlow.

Scoped to what the product actually needs to make its AI features real:
teams and people with skills/history (feeds the capability + assignment
engines), ideas and issues (the work itself), sprints (planning context),
and the governance trail (recommendations + decision history) that every
AI action must write to.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Table
)
from sqlalchemy.orm import relationship
from .database import Base


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, default="")

    employees = relationship("Employee", back_populates="team")


class User(Base):
    """Minimal, structurally-real auth stub \u2014 no password hashing or
    sessions/tokens, since this is a local demo app, not a deployable
    security boundary. Swap for real auth (hashed passwords + JWT/session
    cookies) before deploying this anywhere with real user data."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=True, unique=True)
    role = Column(String, default="Product Manager")  # Admin | Project Manager | Product Manager | Developer | Viewer
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, default="Engineer")
    avatar_color = Column(String, default="#4F5DFF")
    team_id = Column(Integer, ForeignKey("teams.id"))
    status = Column(String, default="available")  # available | absent
    utilization = Column(Float, default=50.0)      # 0-100, % of capacity in use
    skills = Column(JSON, default=dict)             # {"Backend": 91, "API": 96, ...}
    office_id = Column(String, nullable=True, unique=True)
    face_descriptor = Column(JSON, nullable=True)  # 128-length float array from face-api.js; null = not enrolled
    workload_opt_out = Column(Boolean, default=False)  # employee has opted out of workload/recovery-risk analysis

    @property
    def face_enrolled(self) -> bool:
        return self.face_descriptor is not None

    team = relationship("Team", back_populates="employees")
    history = relationship("WorkHistoryItem", back_populates="employee")
    assigned_issues = relationship("Issue", back_populates="assignee")


class WorkHistoryItem(Base):
    """One completed (or in-progress) piece of work, used by the capability
    and assignment engines to compute real success rates instead of guessing."""
    __tablename__ = "work_history"
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    task_type = Column(String)          # Backend | Frontend | QA | DevOps | Mobile | Security
    story_points = Column(Integer, default=3)
    estimated_days = Column(Float, default=3.0)
    actual_days = Column(Float, default=3.0)
    success = Column(Boolean, default=True)
    on_time = Column(Boolean, default=True)
    completed_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="history")


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    key = Column(String, nullable=True, unique=True)  # short code, e.g. "PP" for issue keys like PP-142
    name = Column(String, nullable=False, unique=True)
    description = Column(String, default="")
    goal = Column(String, default="")

    ideas = relationship("Idea", back_populates="project")
    issues = relationship("Issue", back_populates="project")
    sprints = relationship("Sprint", back_populates="project")
    epics = relationship("Epic", back_populates="project")


class Sprint(Base):
    __tablename__ = "sprints"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String, default="planned")  # planned | active | completed
    planned_points = Column(Integer, default=0)
    completed_points = Column(Integer, default=0)
    carry_over_points = Column(Integer, default=0)
    unplanned_points = Column(Integer, default=0)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="sprints")
    issues = relationship("Issue", back_populates="sprint")


class Idea(Base):
    __tablename__ = "ideas"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    problem = Column(Text, default="")
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String, default="New")   # New | Reviewing | Planned | In Delivery | Archived
    team = Column(String, default="")
    goal = Column(String, default="")

    impact = Column(Integer, default=50)
    effort = Column(Integer, default=5)
    confidence = Column(Integer, default=60)
    priority = Column(String, default="MEDIUM")
    priority_reason = Column(Text, default="")

    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="ideas")


class Epic(Base):
    __tablename__ = "epics"
    id = Column(Integer, primary_key=True)
    key = Column(String, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String, default="To Do")  # To Do | In Progress | Done
    color = Column(String, default="#4F5DFF")
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="epics")
    issues = relationship("Issue", back_populates="epic")


issue_labels = Table(
    "issue_labels", Base.metadata,
    Column("issue_id", ForeignKey("issues.id"), primary_key=True),
    Column("label_id", ForeignKey("labels.id"), primary_key=True),
)

issue_components = Table(
    "issue_components", Base.metadata,
    Column("issue_id", ForeignKey("issues.id"), primary_key=True),
    Column("component_id", ForeignKey("components.id"), primary_key=True),
)


class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    color = Column(String, default="#6B7080")


class Component(Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"))


class Issue(Base):
    __tablename__ = "issues"
    id = Column(Integer, primary_key=True)
    key = Column(String, nullable=True, unique=True)  # e.g. "PP-142", generated from project key + id
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    issue_type = Column(String, default="Backend")  # skill/work category: Backend | Frontend | QA | DevOps | Mobile | Security
    item_type = Column(String, default="Task")       # Jira-style kind: Epic-adjacent | Story | Task | Bug | Sub-task | Improvement | Spike
    status = Column(String, default="To Do")        # To Do | In Progress | In Review | Done
    priority = Column(String, default="Medium")
    story_points = Column(Integer, default=3)

    project_id = Column(Integer, ForeignKey("projects.id"))
    sprint_id = Column(Integer, ForeignKey("sprints.id"), nullable=True)
    assignee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    epic_id = Column(Integer, ForeignKey("epics.id"), nullable=True)
    parent_id = Column(Integer, ForeignKey("issues.id"), nullable=True)  # for sub-tasks

    due_date = Column(DateTime, nullable=True)
    blocked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="issues")
    sprint = relationship("Sprint", back_populates="issues")
    assignee = relationship("Employee", back_populates="assigned_issues")
    epic = relationship("Epic", back_populates="issues")
    subtasks = relationship("Issue", backref="parent", remote_side=[id])
    labels = relationship("Label", secondary=issue_labels)
    components = relationship("Component", secondary=issue_components)


class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    issue_id = Column(Integer, ForeignKey("issues.id"))
    author_name = Column(String, default="Anonymous")
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    edited_at = Column(DateTime, nullable=True)


class Watcher(Base):
    __tablename__ = "watchers"
    id = Column(Integer, primary_key=True)
    issue_id = Column(Integer, ForeignKey("issues.id"))
    employee_id = Column(Integer, ForeignKey("employees.id"))


class IssueLink(Base):
    __tablename__ = "issue_links"
    id = Column(Integer, primary_key=True)
    source_issue_id = Column(Integer, ForeignKey("issues.id"))
    target_issue_id = Column(Integer, ForeignKey("issues.id"))
    link_type = Column(String, default="relates to")  # blocks | blocked by | relates to | duplicates


class ActivityLogEntry(Base):
    __tablename__ = "activity_log"
    id = Column(Integer, primary_key=True)
    issue_id = Column(Integer, ForeignKey("issues.id"))
    actor = Column(String, default="System")
    field = Column(String, nullable=False)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Absence(Base):
    """Represented purely as an availability status/window \u2014 no medical or
    personal detail is stored, per the spec's privacy requirement."""
    __tablename__ = "absences"
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    status = Column(String, default="absent")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)


class Attendance(Base):
    """One check-in event. Stores the match distance for biometric
    check-ins for auditability, never the raw face image/video."""
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    method = Column(String, default="manual")  # manual | biometric
    match_distance = Column(Float, nullable=True)
    checked_in_at = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    kind = Column(String, default="general")   # priority | reassignment | risk | duplicate | capacity
    title = Column(String, nullable=False)
    reason = Column(Text, default="")
    evidence = Column(Text, default="")
    confidence = Column(Integer, default=70)
    status = Column(String, default="pending")  # pending | approved | rejected | modified
    related_issue_id = Column(Integer, ForeignKey("issues.id"), nullable=True)
    related_idea_id = Column(Integer, ForeignKey("ideas.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyWorkloadSnapshot(Base):
    """One developer-day activity vector, in the shape the spec's sequence
    model (BiGRU) will eventually consume: completed/active points, active
    task count, after-hours ratio, rest-day flag, etc. Computed daily from
    real WorkHistoryItem + Issue data (never from webcam/emotion/keystroke
    signals) and stored so the 14-28 day rolling window used by the
    Sustainable Workload / Recovery Risk engine doesn't need to be
    recomputed from scratch on every request.

    This table IS the training-ready sequence dataset for a future BiGRU;
    with only a few weeks of seeded history per employee, there isn't
    remotely enough data to train one honestly yet, so today the workload
    engine reads these rows into a deterministic baseline formula instead
    and reports its own status accordingly (see ai/workload_engine.py).
    """
    __tablename__ = "daily_workload_snapshots"
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    date = Column(DateTime, nullable=False)

    completed_story_points = Column(Float, default=0.0)
    active_story_points = Column(Float, default=0.0)
    active_task_count = Column(Integer, default=0)
    context_switch_count = Column(Integer, default=0)     # distinct tasks touched, trailing 7 days
    high_complexity_count = Column(Integer, default=0)     # >=8 story point tasks touched, trailing 7 days
    after_hours_ratio = Column(Float, default=0.0)          # 0-1
    review_queue = Column(Integer, default=0)
    rest_day = Column(Boolean, default=False)
    consecutive_active_days = Column(Integer, default=0)

    load_ratio = Column(Float, default=0.0)
    recovery_risk = Column(Float, default=0.0)  # 0-1, "Sustainable Workload / Recovery Risk" — NOT a medical score

    employee = relationship("Employee")


class AllocationRule(Base):
    """Transparent, manager-configurable symbolic rules (R1-R10) applied by
    the human-aware allocation engine BEFORE scoring. A candidate excluded
    by a hard rule can never become eligible purely by having a high score
    — hard rules always sit above the scoring layer."""
    __tablename__ = "allocation_rules"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, unique=True)   # R1..R10
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    kind = Column(String, default="exclusion")            # exclusion | penalty | review | escalation | info
    enabled = Column(Boolean, default=True)
    threshold = Column(Float, nullable=True)               # editable numeric threshold, where applicable
    weight = Column(Float, nullable=True)                  # editable penalty weight, where applicable


class SimulationRun(Base):
    """Persisted record of a digital-twin/what-if simulation, so the
    Simulator's history is real and re-inspectable rather than ephemeral."""
    __tablename__ = "simulation_runs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    scenario_type = Column(String, default="baseline")
    scenario_json = Column(JSON, default=dict)
    assumptions_json = Column(JSON, default=dict)
    results_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditEvent(Base):
    """Append-only, cryptographically chained audit ledger entry. Every
    consequential AI decision (a recommendation, an approval, a rejection,
    an actual reassignment) gets one of these. `event_hash` is
    SHA-256(CanonicalJSON(payload) + previous_hash); `signature` is an
    Ed25519 signature over `event_hash`. Rows are never updated or deleted
    by application code — only inserted — which is what makes the chain
    meaningful; the DB doesn't enforce this at the schema level (this is a
    prototype, not a production ledger), but every write path goes through
    ai/audit_ledger.py, which never issues UPDATE/DELETE."""
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    event_id = Column(String, unique=True, nullable=False)  # uuid4 hex
    event_type = Column(String, nullable=False)
    payload_json = Column(JSON, default=dict)
    actor_id = Column(String, default="system-ai")
    project_id = Column(Integer, nullable=True)
    task_id = Column(Integer, nullable=True)
    model_version = Column(String, nullable=True)
    feature_schema_version = Column(String, nullable=True)
    rule_ids = Column(JSON, default=list)
    previous_hash = Column(String, nullable=True)
    event_hash = Column(String, nullable=False)
    signature = Column(String, nullable=False)  # hex-encoded Ed25519 signature
    merkle_batch_id = Column(Integer, ForeignKey("merkle_batches.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MerkleBatch(Base):
    """A day's worth of AuditEvent hashes, batched into one Merkle tree so
    the ledger doesn't need one expensive anchor transaction per event.
    `blockchain_tx_id` stays null in this build — see
    ai/fabric_stub.py for why Hyperledger Fabric is an optional,
    unconnected enterprise layer here rather than a live integration."""
    __tablename__ = "merkle_batches"
    id = Column(Integer, primary_key=True)
    batch_date = Column(String, nullable=False, unique=True)  # "YYYY-MM-DD"
    merkle_root = Column(String, nullable=False)
    event_count = Column(Integer, default=0)
    blockchain_tx_id = Column(String, nullable=True)
    platform_identifier = Column(String, default="adaptiveflow-local")
    created_at = Column(DateTime, default=datetime.utcnow)


class DecisionHistoryEntry(Base):
    __tablename__ = "decision_history"
    id = Column(Integer, primary_key=True)
    recommendation = Column(String, nullable=False)
    model = Column(String, default="AdaptiveFlow Engine v1")
    confidence = Column(Integer, default=70)
    evidence = Column(Text, default="")
    human_decision = Column(String, default="Approved")
    outcome = Column(Text, default="")
    predicted_value = Column(String, default="")
    actual_value = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
