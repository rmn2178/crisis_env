"""
tests/test_env.py — Pytest test suite for CrisisEnvironment.
Covers: reset, step (all 5 action types), grader ranges, termination, and determinism.
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.environment import CrisisEnvironment, TOTAL_STEPS
from models import (
    ActionType, ThreatType, CrisisAction, CrisisObservation, StepResult,
    ClassificationPayload, PredictionPayload, AllocationPayload,
    CoordinationPayload, RescuePayload,
)


SEED = 42


@pytest.fixture
def env():
    """Create a fresh environment with fixed seed."""
    e = CrisisEnvironment(seed=SEED)
    return e


# ─────────────────────────────────────────────
# TEST: reset() returns valid observation
# ─────────────────────────────────────────────

def test_reset_returns_valid_observation(env):
    """reset() returns a CrisisObservation with 3 threats and 8 resources."""
    obs = env.reset()
    assert isinstance(obs, CrisisObservation)
    assert len(obs.threats) == 3, f"Expected 3 threats, got {len(obs.threats)}"
    assert len(obs.resources) == 8, f"Expected 8 resources, got {len(obs.resources)}"
    assert obs.time_remaining == TOTAL_STEPS
    assert obs.current_step == 0
    assert len(obs.affected_zones) == 0


# ─────────────────────────────────────────────
# TEST: step() with each of the 5 action types
# ─────────────────────────────────────────────

def test_step_classify(env):
    """step() with CLASSIFY returns StepResult with reward(float) and done(bool)."""
    obs = env.reset()
    threat = obs.threats[0]
    action = CrisisAction(
        action_type=ActionType.CLASSIFY,
        classification=ClassificationPayload(
            threat_id=threat.threat_id,
            predicted_type=threat.threat_type,
            predicted_severity=threat.severity,
        ),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_predict(env):
    """step() with PREDICT returns StepResult with reward(float) and done(bool)."""
    obs = env.reset()
    threat = obs.threats[0]
    action = CrisisAction(
        action_type=ActionType.PREDICT,
        prediction=PredictionPayload(
            threat_id=threat.threat_id,
            predicted_tti=threat.time_to_impact,
            predicted_pop=threat.population_at_risk,
        ),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_allocate(env):
    """step() with ALLOCATE returns StepResult with reward(float) and done(bool)."""
    obs = env.reset()
    threat = obs.threats[0]
    resource = obs.resources[0]
    action = CrisisAction(
        action_type=ActionType.ALLOCATE,
        allocation=AllocationPayload(
            threat_id=threat.threat_id,
            resource_id=resource.resource_id,
        ),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_coordinate(env):
    """step() with COORDINATE returns StepResult with reward(float) and done(bool)."""
    obs = env.reset()
    priority = [t.threat_id for t in obs.threats]
    action = CrisisAction(
        action_type=ActionType.COORDINATE,
        coordination=CoordinationPayload(priority_order=priority),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_rescue(env):
    """step() with RESCUE returns StepResult with reward(float) and done(bool)."""
    obs = env.reset()
    # Rescue action on a zone that doesn't exist yet — should still return valid StepResult
    action = CrisisAction(
        action_type=ActionType.RESCUE,
        rescue=RescuePayload(zone_id=1, rescue_units_to_send=3),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


# ─────────────────────────────────────────────
# TEST: grader scores in [0.0, 1.0]
# ─────────────────────────────────────────────

def test_grader_scores_in_range(env):
    """All 5 grader scores from state() should be in [0.0, 1.0]."""
    obs = env.reset()

    # Do at least one action of each type to produce meaningful grader scores
    threat = obs.threats[0]

    # Classify
    env.step(CrisisAction(
        action_type=ActionType.CLASSIFY,
        classification=ClassificationPayload(
            threat_id=threat.threat_id,
            predicted_type=threat.threat_type,
            predicted_severity=threat.severity,
        ),
    ))

    # Predict
    env.step(CrisisAction(
        action_type=ActionType.PREDICT,
        prediction=PredictionPayload(
            threat_id=threat.threat_id,
            predicted_tti=threat.time_to_impact,
            predicted_pop=threat.population_at_risk,
        ),
    ))

    # Allocate
    resource = obs.resources[0]
    env.step(CrisisAction(
        action_type=ActionType.ALLOCATE,
        allocation=AllocationPayload(
            threat_id=threat.threat_id,
            resource_id=resource.resource_id,
        ),
    ))

    # Coordinate
    priority = [t.threat_id for t in obs.threats]
    env.step(CrisisAction(
        action_type=ActionType.COORDINATE,
        coordination=CoordinationPayload(priority_order=priority),
    ))

    # Rescue (may get a warning since no zone is impacted yet, but score should still be valid)
    env.step(CrisisAction(
        action_type=ActionType.RESCUE,
        rescue=RescuePayload(zone_id=1, rescue_units_to_send=3),
    ))

    state = env.state()

    for score_name in [
        "classification_score",
        "prediction_score",
        "allocation_score",
        "coordination_score",
        "rescue_score",
    ]:
        score = getattr(state, score_name)
        assert 0.0 <= score <= 1.0, f"{score_name}={score} is out of [0.0, 1.0]"


# ─────────────────────────────────────────────
# TEST: done=True after step_count >= 50
# ─────────────────────────────────────────────

def test_done_after_max_steps(env):
    """Episode terminates (done=True) after step_count >= 50."""
    obs = env.reset()
    done = False
    step_count = 0
    priority = [t.threat_id for t in obs.threats]

    while not done and step_count < 60:
        # Use coordinate as a safe repeatable action
        action = CrisisAction(
            action_type=ActionType.COORDINATE,
            coordination=CoordinationPayload(priority_order=priority),
        )
        result = env.step(action)
        done = result.done
        step_count += 1

    assert done, f"Episode did not terminate after {step_count} steps"
    assert step_count <= TOTAL_STEPS, f"Took {step_count} steps, expected <= {TOTAL_STEPS}"


# ─────────────────────────────────────────────
# TEST: same seed produces identical observations
# ─────────────────────────────────────────────

def test_determinism_same_seed():
    """Same seed produces identical observations on two separate resets."""
    env1 = CrisisEnvironment(seed=99)
    env2 = CrisisEnvironment(seed=99)

    obs1 = env1.reset()
    obs2 = env2.reset()

    # Compare threats
    assert len(obs1.threats) == len(obs2.threats)
    for t1, t2 in zip(obs1.threats, obs2.threats):
        assert t1.threat_id == t2.threat_id
        assert t1.threat_type == t2.threat_type
        assert t1.severity == t2.severity
        assert t1.population_at_risk == t2.population_at_risk
        assert t1.time_to_impact == t2.time_to_impact
        assert t1.zone == t2.zone
        assert t1.location_name == t2.location_name

    # Compare resources
    assert len(obs1.resources) == len(obs2.resources)
    for r1, r2 in zip(obs1.resources, obs2.resources):
        assert r1.resource_id == r2.resource_id
        assert r1.resource_type == r2.resource_type
        assert r1.effectiveness == r2.effectiveness


def test_evacuate_action_reduces_population(env):
    """EVACUATE action reduces population_at_risk and returns valid reward."""
    obs = env.reset()
    # Find an active threat
    active = [t for t in obs.threats if t.status.value == "active"]
    assert len(active) > 0
    threat = active[0]
    original_pop = threat.population_at_risk

    action = CrisisAction(
        action_type="evacuate",
        evacuate=__import__("models", fromlist=["EvacuationPayload"]).EvacuationPayload(
            threat_id=threat.threat_id,
            evac_units=5,
        ),
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)

    # Population at risk should have decreased
    updated_threat = next(
        (t for t in result.observation.threats if t.threat_id == threat.threat_id),
        None
    )
    assert updated_threat is not None
    assert updated_threat.population_at_risk < original_pop, (
        "Evacuation should reduce population_at_risk"
    )

    # Evacuation grader should now be > 0
    state = env.state()
    assert state.evacuation_score > 0.0, "Evacuation grader must be > 0 after evacuation"
    assert 0.0 <= state.evacuation_score <= 1.0
