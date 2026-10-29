"""
IdeaUnderstandingEngine + DuplicateDetectionEngine + ImpactPredictionEngine +
EffortPredictionEngine + PriorityEngine.

Deterministic and explainable ("MockPredictionProvider" pattern from the
spec): every score is derived from real signals in the text and in the
existing idea corpus pulled from the database, not random numbers, and
every decision states its reasons. Swappable later for a
RealPredictionProvider backed by an LLM without changing any caller.
"""
import re
from typing import Any, Dict, List
from sqlalchemy.orm import Session
from .. import models

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "is", "are", "in", "on",
    "with", "we", "our", "users", "user", "customers", "customer", "can", "cant",
    "unable", "need", "needs", "should", "would", "want", "wants", "it", "be",
}

DOMAIN_KEYWORDS = {
    "Payments": ["payment", "refund", "invoice", "billing", "checkout", "transaction"],
    "Appointments": ["appointment", "schedule", "reschedule", "booking", "calendar"],
    "Patient Portal": ["patient", "portal", "record", "prescription", "medical"],
    "Security": ["security", "auth", "login", "password", "breach", "encryption"],
    "Analytics": ["analytics", "report", "dashboard", "metric", "insight"],
}

TEAM_BY_DOMAIN = {
    "Payments": "Backend", "Appointments": "Frontend", "Patient Portal": "Mobile",
    "Security": "Security", "Analytics": "Platform Engineering",
}

GOAL_BY_DOMAIN = {
    "Payments": "Reduce operational cost", "Appointments": "Improve customer satisfaction",
    "Patient Portal": "Increase digital adoption", "Security": "Improve security",
    "Analytics": "Improve customer satisfaction",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _detect_domain(text: str) -> str:
    lower = text.lower()
    best, best_hits = "General", 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        hits = sum(1 for k in kws if k in lower)
        if hits > best_hits:
            best, best_hits = domain, hits
    return best


def find_similar(db: Session, text: str, exclude_id: int = None) -> List[Dict]:
    target = _tokens(text)
    results = []
    q = db.query(models.Idea)
    if exclude_id:
        q = q.filter(models.Idea.id != exclude_id)
    for idea in q.all():
        other = _tokens(idea.title + " " + (idea.description or ""))
        sim = _jaccard(target, other)
        pct = round(sim * 100)
        if pct >= 20:
            results.append({"idea_id": idea.id, "title": idea.title, "pct": pct})
    results.sort(key=lambda r: -r["pct"])
    return results[:5]


def predict_impact(text: str, domain: str, similar: List[Dict]) -> Dict:
    lower = text.lower()
    urgency_words = ["unable", "cant", "cannot", "broken", "fail", "critical", "urgent", "blocked"]
    urgency_hits = sum(1 for w in urgency_words if w in lower)
    length_signal = min(20, len(text) // 15)

    customer = min(98, 55 + urgency_hits * 10 + length_signal)
    business = min(96, 50 + (12 if domain in ("Payments", "Security") else 5) + length_signal)
    strategic = min(95, 48 + (10 if domain != "General" else 0) + length_signal)
    overall = round(customer * 0.4 + business * 0.35 + strategic * 0.25)
    confidence = max(45, 90 - len(similar) * 3 - (0 if domain != "General" else 15))
    return {
        "impact": overall, "customer": customer, "business": business,
        "strategic": strategic, "confidence": confidence,
    }


def predict_effort(db: Session, domain: str) -> Dict:
    team = TEAM_BY_DOMAIN.get(domain, "Platform Engineering")
    history = (
        db.query(models.WorkHistoryItem)
        .join(models.Employee)
        .filter(models.Employee.team_id.isnot(None))
        .all()
    )
    relevant = [h.story_points for h in history if h.task_type == ("Backend" if team not in ("Frontend", "Mobile", "QA", "Security") else team)]
    if relevant:
        avg = sum(relevant) / len(relevant)
        low, high = max(1, round(avg - 2)), round(avg + 2)
        predicted = round(avg)
        confidence = min(92, 55 + min(len(relevant), 25))
    else:
        predicted, low, high, confidence = 5, 3, 8, 45
    return {"effort": predicted, "range_low": low, "range_high": high, "confidence": confidence, "sample_size": len(relevant)}


def recommend_priority(impact: int, effort: int, urgency_hits: int, confidence: int) -> Dict:
    score = impact - effort * 3 + urgency_hits * 8
    if score >= 70:
        priority = "HIGH"
    elif score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    reasons = [f"Predicted impact is {impact}/100"]
    if urgency_hits > 0:
        reasons.append("language signals urgency (blocking/critical wording detected)")
    reasons.append(f"estimated effort is {effort} story points")
    if confidence < 60:
        reasons.append("confidence is limited by a small amount of comparable historical data")
    reason_text = "; ".join(reasons) + "."
    return {"priority": priority, "reason": reason_text}


def analyze_idea(db: Session, text: str, exclude_id: int = None) -> Dict[str, Any]:
    domain = _detect_domain(text)
    similar = find_similar(db, text, exclude_id=exclude_id)
    duplicate = similar[0] if similar and similar[0]["pct"] >= 75 else None

    impact_result = predict_impact(text, domain, similar)
    effort_result = predict_effort(db, domain)
    urgency_hits = sum(1 for w in ["unable", "cant", "cannot", "broken", "fail", "critical", "urgent"] if w in text.lower())
    priority_result = recommend_priority(
        impact_result["impact"], effort_result["effort"], urgency_hits, impact_result["confidence"]
    )

    return {
        "problem": text.strip(),
        "feature": f"Capability addressing: {text.strip().rstrip('.')}",
        "user": "Customer" if domain != "Security" else "Internal / Compliance",
        "category": domain,
        "urgency": "High" if urgency_hits > 0 else "Medium",
        "goal": GOAL_BY_DOMAIN.get(domain, "Improve customer satisfaction"),
        "team": TEAM_BY_DOMAIN.get(domain, "Platform Engineering"),
        "impact": impact_result["impact"],
        "impact_breakdown": {k: v for k, v in impact_result.items() if k != "impact"},
        "effort": effort_result["effort"],
        "effort_range": [effort_result["range_low"], effort_result["range_high"]],
        "confidence": min(impact_result["confidence"], effort_result["confidence"]),
        "priority": priority_result["priority"],
        "priority_reason": priority_result["reason"],
        "similar": similar,
        "duplicate": duplicate,
    }
