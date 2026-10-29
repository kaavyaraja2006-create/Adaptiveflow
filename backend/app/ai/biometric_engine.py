"""
BiometricEngine.

Matches a captured face descriptor (128-d embedding produced client-side
by face-api.js) against every enrolled employee using Euclidean distance
\u2014 the standard face-api.js matching approach. Nothing here ever sees a
raw image; only the numeric descriptor, both at enrollment and at
check-in time.
"""
import math
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from .. import models

MATCH_THRESHOLD = 0.6  # face-api.js convention: distances below this are considered the same person


def _distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_best_match(db: Session, descriptor: List[float]) -> Tuple[Optional[models.Employee], Optional[float]]:
    enrolled = db.query(models.Employee).filter(models.Employee.face_descriptor.isnot(None)).all()
    best_emp, best_dist = None, None
    for emp in enrolled:
        d = _distance(descriptor, emp.face_descriptor)
        if best_dist is None or d < best_dist:
            best_emp, best_dist = emp, d
    if best_dist is not None and best_dist <= MATCH_THRESHOLD:
        return best_emp, best_dist
    return None, best_dist
