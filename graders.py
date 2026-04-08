"""Deterministic graders for the Crisis Response Environment.

Three graders — one per difficulty level — all returning float ∈ [0.0, 1.0].
All are stateless: they call ``generate_scenario()`` fresh each invocation to
obtain ground truth and compare against the agent's recorded actions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scenario_generator import generate_scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_score(raw: float) -> float:
    """Clip and round a raw score to [0.0, 1.0]."""
    return round(max(0.0, min(1.0, raw)), 4)


def _PRIORITY_ORDER() -> Dict[str, int]:
    return {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------


def grade_easy(triaged_actions: List[Dict[str, Any]]) -> float:
    """Classification accuracy only.

    Each action that matches the correct priority gets 1 point.
    Score = correct / total.
    """
    if not triaged_actions:
        return 0.0

    correct = 0
    total = len(triaged_actions)
    for action in triaged_actions:
        if action.get("priority_level") == action.get("correct_priority"):
            correct += 1

    return _safe_score(correct / total)


# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------


def grade_medium(triaged_actions: List[Dict[str, Any]]) -> float:
    """Priority accuracy (60 %) + resource allocation accuracy (40 %).

    Priority accuracy:  fraction of actions whose ``priority_level`` matches
    ``correct_priority``.

    Resource accuracy:  fraction of actions whose ``resource_id`` matches
    ``correct_resource`` (only counted when ``resource_id`` is not ``None``
    in either the action or ground truth).
    """
    if not triaged_actions:
        return 0.0

    total = len(triaged_actions)

    # Priority accuracy
    priority_correct = sum(
        1
        for a in triaged_actions
        if a.get("priority_level") == a.get("correct_priority")
    )
    priority_accuracy = priority_correct / total

    # Resource accuracy — count only actions where resource matters
    resource_total = 0
    resource_correct = 0
    for a in triaged_actions:
        gt_resource = a.get("correct_resource")
        agent_resource = a.get("resource_id")
        if gt_resource is not None or agent_resource is not None:
            resource_total += 1
            if agent_resource == gt_resource:
                resource_correct += 1

    resource_accuracy = (resource_correct / resource_total) if resource_total else 0.0

    score = 0.60 * priority_accuracy + 0.40 * resource_accuracy
    return _safe_score(score)


# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------


def _rescue_speed_score(triaged_actions: List[Dict[str, Any]]) -> float:
    """For each threat, score = 1.0 if action received within first 3 steps
    of its appearance, 0.5 if within 5, else 0.0.

    'Appearance step' is approximated as the index of the action in the
    triaged list (i.e. step_number recorded by the environment). The first
    action for each threat is what matters.
    """
    if not triaged_actions:
        return 0.0

    # First appearance step per threat
    first_step: Dict[str, int] = {}
    for a in triaged_actions:
        tid = a.get("threat_id", "")
        step = a.get("step_number", 999)
        if tid not in first_step:
            first_step[tid] = step

    # Score each threat
    total = 0.0
    for _tid, step in first_step.items():
        if step <= 3:
            total += 1.0
        elif step <= 5:
            total += 0.5
        else:
            total += 0.0

    return total / len(first_step) if first_step else 0.0


def grade_hard(triaged_actions: List[Dict[str, Any]]) -> float:
    """Priority + resource + rescue speed − CRITICAL miss penalty.

    Formula:
        raw = 0.40 * priority_accuracy
            + 0.30 * resource_accuracy
            + 0.30 * rescue_speed_score
            − critical_miss_penalty   (0.20 per CRITICAL threat missed)
    """
    if not triaged_actions:
        return 0.0

    total = len(triaged_actions)

    # Priority accuracy
    priority_correct = sum(
        1
        for a in triaged_actions
        if a.get("priority_level") == a.get("correct_priority")
    )
    priority_accuracy = priority_correct / total

    # Resource accuracy
    resource_total = 0
    resource_correct = 0
    for a in triaged_actions:
        gt_resource = a.get("correct_resource")
        agent_resource = a.get("resource_id")
        if gt_resource is not None or agent_resource is not None:
            resource_total += 1
            if agent_resource == gt_resource:
                resource_correct += 1
    resource_accuracy = (resource_correct / resource_total) if resource_total else 0.0

    # Rescue speed
    rescue_speed = _rescue_speed_score(triaged_actions)

    # CRITICAL miss penalty — find CRITICAL threats that were not given
    # priority_level == CRITICAL by the agent
    critical_misses = 0
    for a in triaged_actions:
        if a.get("correct_priority") == "CRITICAL" and a.get("priority_level") != "CRITICAL":
            critical_misses += 1
    critical_miss_penalty = 0.20 * critical_misses

    raw = (
        0.40 * priority_accuracy
        + 0.30 * resource_accuracy
        + 0.30 * rescue_speed
        - critical_miss_penalty
    )
    return _safe_score(raw)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "name": "Single-Threat Classification",
        "difficulty": "easy",
        "description": "5 clear threats, one at a time. No resource conflicts. Expected score: ~0.80",
        "grader": grade_easy,
        "scenario_size": 5,
    },
    "medium": {
        "id": "medium",
        "name": "Multi-Threat Coordination",
        "difficulty": "medium",
        "description": "10 mixed threats, hidden cascade. Prioritization required. Expected score: ~0.55",
        "grader": grade_medium,
        "scenario_size": 10,
    },
    "hard": {
        "id": "hard",
        "name": "Full Lifecycle Rescue Optimization",
        "difficulty": "hard",
        "description": "15 threats, resource scarcity, rescue phase. CRITICAL miss penalty. Expected score: ~0.30",
        "grader": grade_hard,
        "scenario_size": 15,
    },
}
