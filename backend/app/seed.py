"""
Seeds a realistic NovaCare Technologies demo org: teams, employees with
skills and multi-year-equivalent work history, projects, sprints with real
carry-over/unplanned signals, issues, ideas, and one pre-baked absence
scenario \u2014 so every AI engine has real data to compute against the first
time you open the app.
"""
import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from . import models
from .ai import workload_engine, rules_engine

random.seed(42)

TEAMS = ["Platform Engineering", "Frontend", "Backend", "Mobile", "QA", "Security", "DevOps", "Product"]

TASK_TYPES = ["Backend", "Frontend", "QA", "DevOps", "Mobile", "Security"]

EMPLOYEE_NAMES = [
    "Priya Sharma", "Rahul Mehta", "Sneha Iyer", "Karthik Rao", "Ananya Gupta",
    "Vikram Nair", "Divya Menon", "Arjun Reddy", "Kavya Pillai", "Rohan Kapoor",
    "Meera Krishnan", "Aditya Verma", "Nisha Pillai", "Sanjay Kumar", "Pooja Desai",
    "Farhan Ali", "Ritu Chawla", "Manoj Tiwari", "Swati Joshi", "Aakash Bansal",
]

AVATAR_COLORS = ["#4F5DFF", "#1F9D6E", "#C98A1B", "#D14343", "#3E6AE1", "#8B5CF6"]

PROJECTS = [
    ("Patient Portal", "PP", "Self-service portal for patients", "Increase digital adoption"),
    ("Appointment Platform", "AP", "Scheduling and reminders", "Improve customer satisfaction"),
    ("Payment Platform", "PAY", "Billing, invoicing and refunds", "Reduce operational cost"),
    ("Analytics Platform", "AN", "Internal reporting and insights", "Improve customer satisfaction"),
    ("Security & Compliance", "SEC", "Auth, audit and data protection", "Improve security"),
]

IDEA_SEEDS = [
    ("Customers are unable to reschedule appointments online", "Appointment Platform"),
    ("Patients want to download prescriptions as PDF", "Patient Portal"),
    ("Refunds take too long to process after cancellation", "Payment Platform"),
    ("Support team wants a dashboard of recurring complaints", "Analytics Platform"),
    ("We need two-factor authentication for clinician logins", "Security & Compliance"),
    ("Appointment reminders should go out via SMS not just email", "Appointment Platform"),
    ("Billing invoices are confusing and generate support tickets", "Payment Platform"),
    ("Patients cannot cancel appointments after business hours", "Appointment Platform"),
    ("We should detect duplicate patient records automatically", "Patient Portal"),
    ("Analytics dashboard is too slow when filtering by month", "Analytics Platform"),
    ("Add audit logging for all admin actions", "Security & Compliance"),
    ("Patients want appointment booking directly from mobile app", "Patient Portal"),
]

ISSUE_TITLES_BY_TYPE = {
    "Backend": ["Refund API", "Payment validation service", "Auth token refresh", "Appointment API rate limiting",
                "Patient record deduplication job", "Invoice generation service", "Webhook retry logic"],
    "Frontend": ["Reschedule appointment UI", "Invoice detail page", "Patient portal dashboard redesign",
                 "Appointment calendar component", "Notification preferences screen"],
    "QA": ["Regression suite for payments", "Appointment booking E2E tests", "Security penetration test pass",
           "Load test for analytics queries"],
    "DevOps": ["CI pipeline for mobile app", "Blue-green deploy for payment service", "Log aggregation setup"],
    "Mobile": ["Mobile appointment booking flow", "Push notifications for reminders", "Biometric login"],
    "Security": ["Two-factor authentication rollout", "Audit log for admin actions", "Encrypt data at rest"],
}


def _random_skills():
    return {t: random.randint(45, 97) for t in TASK_TYPES}


def seed(db: Session):
    if db.query(models.Team).first():
        return  # already seeded

    teams = {}
    for name in TEAMS:
        t = models.Team(name=name, description=f"{name} team at NovaCare Technologies")
        db.add(t)
        teams[name] = t
    db.commit()

    team_cycle = TEAMS
    employees = []
    for i, name in enumerate(EMPLOYEE_NAMES):
        team_name = team_cycle[i % len(team_cycle)]
        emp = models.Employee(
            name=name,
            role=random.choice(["Engineer", "Senior Engineer", "Lead Engineer", "QA Engineer"]),
            avatar_color=random.choice(AVATAR_COLORS),
            team_id=teams[team_name].id,
            status="available",
            utilization=random.randint(25, 90),
            skills=_random_skills(),
            office_id=f"NC-{1000 + i}",
        )
        db.add(emp)
        employees.append(emp)
    db.commit()

    # demo signal for the Sustainable Workload / Recovery Risk engine: give
    # one employee a clearly high utilization so their recovery risk shows
    # up as "elevated"/"severe" in the demo, and let one employee opt out
    # of workload analysis entirely (per the spec's opt-out requirement).
    karthik = next((e for e in employees if e.name == "Karthik Rao"), None)
    if karthik:
        karthik.utilization = 93
    meera = next((e for e in employees if e.name == "Meera Krishnan"), None)
    if meera:
        meera.workload_opt_out = True
    db.commit()

    # realistic work history so success rates aren't all defaults
    for emp in employees:
        n_tasks = random.randint(6, 16)
        for _ in range(n_tasks):
            task_type = random.choice(TASK_TYPES)
            skill = emp.skills.get(task_type, 50)
            success_chance = min(0.97, 0.4 + skill / 130)
            success = random.random() < success_chance
            on_time = random.random() < (success_chance - 0.05)
            points = random.choice([1, 2, 3, 5, 8, 13])
            est = points * random.uniform(0.7, 1.3)
            actual = est * (random.uniform(0.85, 1.15) if on_time else random.uniform(1.15, 1.6))
            db.add(models.WorkHistoryItem(
                employee_id=emp.id, task_type=task_type, story_points=points,
                estimated_days=round(est, 1), actual_days=round(actual, 1),
                success=success, on_time=on_time,
                completed_at=datetime.utcnow() - timedelta(days=random.randint(5, 300)),
            ))
    db.commit()

    projects = {}
    for name, key, desc, goal in PROJECTS:
        p = models.Project(name=name, key=key, description=desc, goal=goal)
        db.add(p)
        projects[name] = p
    db.commit()

    sprints = []
    for name, project in projects.items():
        for n in range(1, 4):
            planned = random.randint(30, 55)
            completed = round(planned * random.uniform(0.6, 0.95))
            carry = max(0, planned - completed - random.randint(0, 5))
            unplanned = random.randint(0, 10)
            # sprint 3 is the active one; give every sprint a real 14-day
            # window so the digital twin's sprint-miss probability has an
            # actual deadline to simulate against, not just a proxy.
            days_since_sprint3_start = (3 - n) * 14
            start_date = datetime.utcnow() - timedelta(days=5 + days_since_sprint3_start)
            end_date = start_date + timedelta(days=14)
            s = models.Sprint(
                name=f"Sprint {n}", project_id=project.id,
                status="completed" if n < 3 else "active",
                planned_points=planned, completed_points=completed,
                carry_over_points=carry, unplanned_points=unplanned,
                start_date=start_date, end_date=end_date,
            )
            db.add(s)
            sprints.append(s)
    db.commit()

    # issues, spread across projects/types, most assigned, some blocked
    all_issues = []
    for name, project in projects.items():
        proj_sprints = [s for s in sprints if s.project_id == project.id]
        for task_type, titles in ISSUE_TITLES_BY_TYPE.items():
            for title in random.sample(titles, k=min(2, len(titles))):
                assignee = random.choice([e for e in employees])
                status = random.choice(["To Do", "In Progress", "In Review", "Done", "Done", "In Progress"])
                issue = models.Issue(
                    title=f"{title} \u2014 {name}",
                    description=f"Work item for {name}: {title}.",
                    issue_type=task_type,
                    item_type=random.choice(["Task", "Task", "Story", "Story", "Bug", "Improvement"]),
                    status=status,
                    priority=random.choice(["Low", "Medium", "High", "Critical"]),
                    story_points=random.choice([1, 2, 3, 5, 8]),
                    project_id=project.id,
                    sprint_id=random.choice(proj_sprints).id if proj_sprints else None,
                    assignee_id=assignee.id,
                    blocked=random.random() < 0.12,
                )
                db.add(issue)
                all_issues.append(issue)
    db.commit()
    for issue in all_issues:
        issue.key = f"{issue.project.key or 'ISS'}-{issue.id}"
    db.commit()

    # epics per project, with a handful of issues rolled up under each
    epics_by_project = {}
    for name, project in projects.items():
        proj_issues = [i for i in all_issues if i.project_id == project.id]
        for n, title in enumerate([f"{name} Foundations", f"{name} Improvements"], start=1):
            epic = models.Epic(title=title, project_id=project.id, color=random.choice(["#4F5DFF", "#1F9D6E", "#C98A1B"]))
            db.add(epic)
            db.commit()
            epic.key = f"{project.key or 'EPIC'}-E{epic.id}"
            epics_by_project.setdefault(project.id, []).append(epic)
        db.commit()
        # attach roughly half this project's issues to one of its epics
        for issue in random.sample(proj_issues, k=min(len(proj_issues) // 2, len(proj_issues))):
            issue.epic_id = random.choice(epics_by_project[project.id]).id
    db.commit()

    # labels and components, applied to a subset of issues
    label_names = ["urgent", "customer-request", "technical-debt", "security", "quick-win"]
    labels = []
    for n in label_names:
        l = models.Label(name=n, color=random.choice(["#D14343", "#4F5DFF", "#C98A1B", "#1F9D6E"]))
        db.add(l)
        labels.append(l)
    db.commit()

    components_by_project = {}
    for name, project in projects.items():
        comp_names = random.sample(["Frontend", "Backend", "Database", "Auth", "Mobile"], k=3)
        components_by_project[project.id] = []
        for cn in comp_names:
            c = models.Component(name=cn, project_id=project.id)
            db.add(c)
            components_by_project[project.id].append(c)
    db.commit()

    for issue in random.sample(all_issues, k=len(all_issues) // 2):
        issue.labels = random.sample(labels, k=random.randint(1, 2))
        proj_components = components_by_project.get(issue.project_id, [])
        if proj_components:
            issue.components = random.sample(proj_components, k=min(1, len(proj_components)))
    db.commit()

    # a handful of realistic comments on a subset of issues
    comment_bodies = [
        "Confirmed this is reproducible in staging.",
        "Blocked on the upstream API change \u2014 following up with platform team.",
        "Looks good, moving to review.",
        "Can we get a design review before starting this?",
        "This will need a migration script for existing records.",
    ]
    for issue in random.sample(all_issues, k=len(all_issues) // 3):
        for _ in range(random.randint(1, 2)):
            db.add(models.Comment(
                issue_id=issue.id,
                author_name=random.choice(employees).name,
                body=random.choice(comment_bodies),
            ))
    db.commit()

    # ideas, seeded with real text so duplicate detection / similarity have signal
    for text, project_name in IDEA_SEEDS:
        db.add(models.Idea(
            title=text, description="", problem=text,
            project_id=projects[project_name].id,
            team="Platform Engineering", goal=projects[project_name].goal,
            impact=random.randint(50, 92), effort=random.randint(3, 10),
            confidence=random.randint(55, 88), priority=random.choice(["LOW", "MEDIUM", "HIGH"]),
            priority_reason="Seeded historical idea.",
        ))
    db.commit()

    # pre-baked absence scenario: Rahul Mehta has 3 active tasks and is NOT
    # marked absent yet, so the demo can trigger the flow live.
    rahul = next((e for e in employees if e.name == "Rahul Mehta"), employees[1])
    active_issues = [i for i in all_issues if i.status != "Done"][:3]
    for issue in active_issues:
        issue.assignee_id = rahul.id
    db.commit()

    # pre-baked workload/recovery-risk demo scenario: give Karthik Rao a
    # heavy active load (several high-point tasks) on top of his high
    # utilization, so the Sustainable Workload / Recovery Risk engine shows
    # a clearly "elevated"/"severe" band for at least one real person in
    # the demo, rather than everyone looking artificially fine.
    if karthik:
        overload_issues = [i for i in all_issues if i.status != "Done" and i.assignee_id != rahul.id][:6]
        for issue in overload_issues:
            issue.assignee_id = karthik.id
            issue.story_points = max(issue.story_points or 3, random.choice([8, 13]))
        db.commit()

    # demo user personas (auth stub \u2014 no passwords, just a "log in as" picker)
    priya = next((e for e in employees if e.name == "Priya Sharma"), employees[0])
    db.add_all([
        models.User(name="Admin User", email="admin@novacare.demo", role="Admin"),
        models.User(name="Project Manager", email="pm@novacare.demo", role="Project Manager"),
        models.User(name="Priya Sharma", email="priya@novacare.demo", role="Developer", employee_id=priya.id),
        models.User(name="Viewer", email="viewer@novacare.demo", role="Viewer"),
    ])
    db.commit()

    # pre-generate the 21-day workload/recovery-risk activity window for
    # every employee (except those opted out) so the demo is immediately
    # populated rather than lazily computing on first API call, and seed
    # the default symbolic rule set (R1-R10).
    for emp in employees:
        if not emp.workload_opt_out:
            workload_engine.ensure_snapshots(db, emp)
    rules_engine.ensure_default_rules(db)

    # Build the duration-forecasting model (Digital Twin — Phase 4-7).
    # There's real WorkHistoryItem data by this point, so this is a
    # genuine fit (NGBoost if installed, else the honest empirical
    # baseline) — never a fabricated "trained" status.
    from .ai import forecasting_engine
    forecasting_engine.train(db)
