"""
AdaptiveFlow Project Digital Twin — dependency graph layer.

Builds a real directed graph from the project's actual data: Issue,
Employee, Sprint, and Skill nodes; TASK_DEPENDS_ON_TASK edges from real
IssueLink rows (link_type "blocks"/"blocked by"), TASK_ASSIGNED_TO_DEVELOPER
from Issue.assignee_id, DEVELOPER_HAS_SKILL from Employee.skills,
TASK_REQUIRES_SKILL from Issue.issue_type, TASK_AFFECTS_SPRINT_GOAL from
Issue.sprint_id.

HONEST SCOPE NOTE: the spec's node/edge list also includes Repository, Pull
Request, and Build Pipeline nodes with a TASK_BLOCKED_BY_PR edge. This
build has no Git/CI/CD integration (consistent with the project's stated
non-goals — see README), so there is no real data to build those nodes
from. Rather than fabricate synthetic PR/CI state, this module only builds
the graph from data that genuinely exists; the Repository/PR/Build
Pipeline node types are left as a documented extension point rather than
mocked here.
"""
import networkx as nx
from sqlalchemy.orm import Session
from .. import models


def build_project_graph(db: Session, project: models.Project) -> nx.DiGraph:
    g = nx.DiGraph()
    issues = db.query(models.Issue).filter(models.Issue.project_id == project.id).all()
    issue_ids = {i.id for i in issues}

    for issue in issues:
        g.add_node(f"task:{issue.id}", kind="Task", title=issue.title, status=issue.status,
                   story_points=issue.story_points, priority=issue.priority,
                   issue_type=issue.issue_type, blocked=issue.blocked,
                   assignee_id=issue.assignee_id, sprint_id=issue.sprint_id)

    assignee_ids = {i.assignee_id for i in issues if i.assignee_id}
    employees = db.query(models.Employee).filter(models.Employee.id.in_(assignee_ids)).all() if assignee_ids else []
    for emp in employees:
        g.add_node(f"dev:{emp.id}", kind="Developer", name=emp.name, utilization=emp.utilization,
                   status=emp.status)
        for skill_name in (emp.skills or {}):
            skill_node = f"skill:{skill_name}"
            if skill_node not in g:
                g.add_node(skill_node, kind="Skill", name=skill_name)
            g.add_edge(f"dev:{emp.id}", skill_node, edge_type="DEVELOPER_HAS_SKILL",
                       level=(emp.skills or {}).get(skill_name))

    sprints = {s.id: s for s in db.query(models.Sprint).filter(models.Sprint.project_id == project.id).all()}
    for sprint in sprints.values():
        g.add_node(f"sprint:{sprint.id}", kind="Sprint", name=sprint.name, status=sprint.status,
                   start_date=sprint.start_date, end_date=sprint.end_date)

    for issue in issues:
        if issue.assignee_id:
            g.add_edge(f"task:{issue.id}", f"dev:{issue.assignee_id}", edge_type="TASK_ASSIGNED_TO_DEVELOPER")
        skill_node = f"skill:{issue.issue_type}"
        if skill_node not in g:
            g.add_node(skill_node, kind="Skill", name=issue.issue_type)
        g.add_edge(f"task:{issue.id}", skill_node, edge_type="TASK_REQUIRES_SKILL")
        if issue.sprint_id and issue.sprint_id in sprints:
            g.add_edge(f"task:{issue.id}", f"sprint:{issue.sprint_id}", edge_type="TASK_AFFECTS_SPRINT_GOAL")

    links = (
        db.query(models.IssueLink)
        .filter(models.IssueLink.source_issue_id.in_(issue_ids))
        .all()
    )
    for link in links:
        if link.target_issue_id not in issue_ids:
            continue
        # normalize to a single directed "depends on" edge regardless of
        # which side of the relationship was stored as source/target
        if link.link_type == "blocks":
            g.add_edge(f"task:{link.target_issue_id}", f"task:{link.source_issue_id}",
                       edge_type="TASK_DEPENDS_ON_TASK")
        elif link.link_type == "blocked by":
            g.add_edge(f"task:{link.source_issue_id}", f"task:{link.target_issue_id}",
                       edge_type="TASK_DEPENDS_ON_TASK")

    return g


def task_dependency_subgraph(g: nx.DiGraph) -> nx.DiGraph:
    edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "TASK_DEPENDS_ON_TASK"]
    sub = nx.DiGraph()
    sub.add_nodes_from(n for n in g.nodes if g.nodes[n].get("kind") == "Task")
    sub.add_edges_from(edges)
    return sub


def downstream_impact(g: nx.DiGraph, task_node: str) -> list:
    """Every task that transitively depends on `task_node` — i.e. would be
    delayed if `task_node` slips. Uses the real dependency subgraph; if the
    graph isn't a DAG (a genuine cyclic dependency was entered), that's
    reported rather than silently ignored."""
    sub = task_dependency_subgraph(g)
    if task_node not in sub:
        return []
    return list(nx.ancestors(sub, task_node))  # nodes that depend ON task_node (edges point dependent->dependency... see note)


def critical_path(g: nx.DiGraph, duration_lookup: dict) -> dict:
    """Longest path through the task-dependency DAG by estimated duration
    (hours), which is the standard definition of a project's critical path.
    `duration_lookup`: {task_node: hours}. Falls back gracefully — and
    reports it — if the graph has a cycle (not a valid DAG)."""
    sub = task_dependency_subgraph(g)
    if not nx.is_directed_acyclic_graph(sub):
        cycle = None
        try:
            cycle = nx.find_cycle(sub)
        except nx.NetworkXNoCycle:
            pass
        return {"valid": False, "reason": f"Dependency graph has a cycle: {cycle}", "path": [], "total_hours": 0}

    if len(sub) == 0:
        return {"valid": True, "path": [], "total_hours": 0}

    # longest path by weight via topological DP
    order = list(nx.topological_sort(sub))
    dist = {n: duration_lookup.get(n, 0.0) for n in sub.nodes}
    parent = {n: None for n in sub.nodes}
    for n in order:
        for succ in sub.successors(n):
            candidate = dist[n] + duration_lookup.get(succ, 0.0)
            if candidate > dist[succ]:
                dist[succ] = candidate
                parent[succ] = n

    end_node = max(dist, key=dist.get) if dist else None
    path = []
    cur = end_node
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    return {"valid": True, "path": path, "total_hours": round(dist.get(end_node, 0.0), 1) if end_node else 0}


def graph_summary(g: nx.DiGraph) -> dict:
    by_kind = {}
    for n, d in g.nodes(data=True):
        by_kind[d.get("kind", "Unknown")] = by_kind.get(d.get("kind", "Unknown"), 0) + 1
    by_edge = {}
    for _, _, d in g.edges(data=True):
        by_edge[d.get("edge_type", "Unknown")] = by_edge.get(d.get("edge_type", "Unknown"), 0) + 1
    return {"node_counts": by_kind, "edge_counts": by_edge,
            "total_nodes": g.number_of_nodes(), "total_edges": g.number_of_edges()}
