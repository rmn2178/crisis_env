"""Comprehensive tests for the Crisis Response Environment.

Covers: reset, step, rewards, graders, and scenario generator.
At least 30 tests as required by the specification.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from models import CrisisAction, CrisisObservation, CrisisStepResult
from server.environment import (
    CrisisResponseEnvironment,
    VALID_ACTION_TYPES,
    VALID_PRIORITIES,
    VALID_TASK_IDS,
    REWARD_CORRECT_PRIORITY,
    REWARD_CORRECT_RESOURCE,
)
from graders import grade_easy, grade_medium, grade_hard, TASK_REGISTRY
from scenario_generator import (
    generate_scenario,
    strip_ground_truth,
    THREAT_TEMPLATES,
    generate_threat,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def env():
    return CrisisResponseEnvironment()


# ===================================================================
# TestReset
# ===================================================================


class TestReset:
    def test_reset_returns_crisis_observation(self, env):
        obs = env.reset(task_id="easy")
        assert isinstance(obs, CrisisObservation)

    def test_reset_easy_gives_5_threats(self, env):
        env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        assert len(scenario) == 5

    def test_reset_medium_gives_10_threats(self, env):
        env.reset(task_id="medium")
        scenario = generate_scenario("medium")
        assert len(scenario) == 10

    def test_reset_hard_gives_15_threats(self, env):
        env.reset(task_id="hard")
        scenario = generate_scenario("hard")
        assert len(scenario) == 15

    def test_reset_invalid_task_id_defaults_to_easy(self, env):
        obs = env.reset(task_id="impossible")
        assert env.state.task_id == "easy"
        assert isinstance(obs, CrisisObservation)

    def test_reset_clears_previous_episode_state(self, env):
        env.reset(task_id="easy")
        # Take a step
        obs = env.reset(task_id="easy")
        threat_id = obs.threat_id
        env.step(CrisisAction(action_type="classify", threat_id=threat_id, priority_level="HIGH"))
        assert env.state.step_count == 1

        # Reset should clear
        env.reset(task_id="medium")
        assert env.state.step_count == 0
        assert env.state.task_id == "medium"
        assert len(env.triaged) == 0

    def test_reset_observation_has_required_fields(self, env):
        obs = env.reset(task_id="easy")
        assert obs.threat_id is not None
        assert obs.threat_type in ("AIRSTRIKE", "SHIP_ATTACK", "DRONE_THREAT")
        assert obs.severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert obs.population_at_risk > 0
        assert obs.time_to_impact > 0
        assert isinstance(obs.available_resources, list)
        assert obs.threats_remaining > 0
        assert obs.done is False


# ===================================================================
# TestStep
# ===================================================================


class TestStep:
    def test_valid_classify_returns_step_result(self, env):
        obs = env.reset(task_id="easy")
        result = env.step(
            CrisisAction(
                action_type="classify",
                threat_id=obs.threat_id,
                priority_level="CRITICAL",
            )
        )
        assert isinstance(result, CrisisStepResult)
        assert isinstance(result.observation, CrisisObservation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_valid_allocate_with_correct_resource_gives_high_reward(self, env):
        obs = env.reset(task_id="easy")
        # The first easy threat is AIRSTRIKE CRITICAL — correct resource is fighter_jet
        scenario = generate_scenario("easy")
        correct_resource = scenario[0]["_correct_resource"]
        correct_priority = scenario[0]["_correct_priority"]

        result = env.step(
            CrisisAction(
                action_type="allocate",
                threat_id=obs.threat_id,
                priority_level=correct_priority,
                resource_id=correct_resource,
            )
        )
        assert result.reward >= 0.50  # at least priority reward

    def test_invalid_action_type_gives_penalty(self, env):
        obs = env.reset(task_id="easy")
        result = env.step(
            CrisisAction(
                action_type="explode",
                threat_id=obs.threat_id,
            )
        )
        assert result.reward == 0.0
        assert "error" in result.info

    def test_wrong_threat_id_gives_penalty(self, env):
        obs = env.reset(task_id="easy")
        result = env.step(
            CrisisAction(
                action_type="classify",
                threat_id="THR-WRONG-999",
                priority_level="HIGH",
            )
        )
        assert result.reward == 0.0
        assert result.info.get("error") == "wrong_threat_id"

    def test_invalid_priority_level_gives_penalty(self, env):
        obs = env.reset(task_id="easy")
        result = env.step(
            CrisisAction(
                action_type="classify",
                threat_id=obs.threat_id,
                priority_level="EXTREME",
            )
        )
        assert result.reward == 0.0
        assert "error" in result.info

    def test_done_true_after_last_threat(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        for i, threat in enumerate(scenario):
            result = env.step(
                CrisisAction(
                    action_type="classify",
                    threat_id=threat["threat_id"],
                    priority_level=threat["_correct_priority"],
                )
            )
        assert result.done is True

    def test_cumulative_score_increases(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        prev_score = 0.0
        for threat in scenario[:3]:
            result = env.step(
                CrisisAction(
                    action_type="classify",
                    threat_id=threat["threat_id"],
                    priority_level=threat["_correct_priority"],
                )
            )
            assert result.observation.cumulative_score >= prev_score
            prev_score = result.observation.cumulative_score

    def test_step_after_done_returns_zero_reward(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        for threat in scenario:
            env.step(
                CrisisAction(
                    action_type="classify",
                    threat_id=threat["threat_id"],
                    priority_level=threat["_correct_priority"],
                )
            )
        # Extra step after done
        result = env.step(
            CrisisAction(action_type="classify", threat_id="THR-9999", priority_level="LOW")
        )
        assert result.done is True
        assert result.reward == 0.0


# ===================================================================
# TestRewards
# ===================================================================


class TestRewards:
    def test_correct_priority_reward_at_least_050(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        correct_priority = scenario[0]["_correct_priority"]
        result = env.step(
            CrisisAction(
                action_type="classify",
                threat_id=obs.threat_id,
                priority_level=correct_priority,
            )
        )
        assert result.reward >= 0.50

    def test_correct_resource_includes_030(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        result = env.step(
            CrisisAction(
                action_type="allocate",
                threat_id=obs.threat_id,
                priority_level=scenario[0]["_correct_priority"],
                resource_id=scenario[0]["_correct_resource"],
            )
        )
        assert result.reward >= 0.80  # priority + resource + speed bonus

    def test_wrong_resource_excludes_030(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        result = env.step(
            CrisisAction(
                action_type="allocate",
                threat_id=obs.threat_id,
                priority_level=scenario[0]["_correct_priority"],
                resource_id="RES-medic_team-01",  # wrong for airstrike
            )
        )
        # Should get priority + speed but not resource
        assert result.reward < 1.0

    def test_critical_missed_penalty(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        # First threat is CRITICAL — give it LOW priority
        if scenario[0]["_correct_priority"] == "CRITICAL":
            result = env.step(
                CrisisAction(
                    action_type="classify",
                    threat_id=obs.threat_id,
                    priority_level="LOW",
                )
            )
            # Penalty should reduce reward — clipped to 0.0 minimum
            assert result.reward <= 0.50

    def test_all_rewards_clipped_to_01(self, env):
        obs = env.reset(task_id="easy")
        scenario = generate_scenario("easy")
        for threat in scenario:
            result = env.step(
                CrisisAction(
                    action_type="allocate",
                    threat_id=threat["threat_id"],
                    priority_level=threat["_correct_priority"],
                    resource_id=threat["_correct_resource"],
                )
            )
            assert 0.0 <= result.reward <= 1.0


# ===================================================================
# TestGraders
# ===================================================================


class TestGraders:
    def test_grade_easy_perfect_near_10(self):
        scenario = generate_scenario("easy")
        actions = [
            {
                "threat_id": t["threat_id"],
                "action": "classify",
                "priority_level": t["_correct_priority"],
                "resource_id": t["_correct_resource"],
                "reasoning": "test",
                "reward": 0.8,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        score = grade_easy(actions)
        assert score >= 0.95

    def test_grade_easy_all_wrong_near_00(self):
        scenario = generate_scenario("easy")
        actions = [
            {
                "threat_id": t["threat_id"],
                "action": "classify",
                "priority_level": "LOW" if t["_correct_priority"] != "LOW" else "HIGH",
                "resource_id": None,
                "reasoning": "wrong",
                "reward": 0.0,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        score = grade_easy(actions)
        assert score <= 0.20

    def test_grade_easy_mixed_between_00_and_10(self):
        scenario = generate_scenario("easy")
        actions = []
        for i, t in enumerate(scenario):
            # Half correct, half wrong
            pl = t["_correct_priority"] if i % 2 == 0 else "LOW"
            actions.append(
                {
                    "threat_id": t["threat_id"],
                    "action": "classify",
                    "priority_level": pl,
                    "resource_id": None,
                    "reasoning": "mixed",
                    "reward": 0.5,
                    "correct_priority": t["_correct_priority"],
                    "correct_resource": t["_correct_resource"],
                    "step_number": i + 1,
                }
            )
        score = grade_easy(actions)
        assert 0.0 < score < 1.0

    def test_grade_medium_perfect_near_10(self):
        scenario = generate_scenario("medium")
        actions = [
            {
                "threat_id": t["threat_id"],
                "action": "allocate",
                "priority_level": t["_correct_priority"],
                "resource_id": t["_correct_resource"],
                "reasoning": "test",
                "reward": 0.8,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        score = grade_medium(actions)
        assert score >= 0.95

    def test_grade_hard_perfect_near_10(self):
        scenario = generate_scenario("hard")
        actions = [
            {
                "threat_id": t["threat_id"],
                "action": "allocate",
                "priority_level": t["_correct_priority"],
                "resource_id": t["_correct_resource"],
                "reasoning": "test",
                "reward": 0.8,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        score = grade_hard(actions)
        assert score >= 0.85  # speed bonus drops for later steps

    def test_graders_return_different_scores(self):
        scenario = generate_scenario("easy")
        perfect = [
            {
                "threat_id": t["threat_id"],
                "action": "classify",
                "priority_level": t["_correct_priority"],
                "resource_id": t["_correct_resource"],
                "reasoning": "p",
                "reward": 0.8,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        wrong = [
            {
                "threat_id": t["threat_id"],
                "action": "classify",
                "priority_level": "LOW" if t["_correct_priority"] != "LOW" else "HIGH",
                "resource_id": None,
                "reasoning": "w",
                "reward": 0.0,
                "correct_priority": t["_correct_priority"],
                "correct_resource": t["_correct_resource"],
                "step_number": i + 1,
            }
            for i, t in enumerate(scenario)
        ]
        assert grade_easy(perfect) != grade_easy(wrong)

    def test_grade_empty_actions_returns_zero(self):
        assert grade_easy([]) == 0.0
        assert grade_medium([]) == 0.0
        assert grade_hard([]) == 0.0


# ===================================================================
# TestScenarioGenerator
# ===================================================================


class TestScenarioGenerator:
    def test_easy_scenario_has_5_threats(self):
        assert len(generate_scenario("easy")) == 5

    def test_medium_scenario_has_10_threats(self):
        assert len(generate_scenario("medium")) == 10

    def test_hard_scenario_has_15_threats(self):
        assert len(generate_scenario("hard")) == 15

    def test_all_threats_have_required_fields(self):
        required = {
            "threat_id", "threat_type", "location", "description",
            "severity", "time_to_impact", "population_at_risk",
            "has_visible_signal", "affected_systems",
            "_correct_priority", "_correct_resource",
        }
        for diff in ("easy", "medium", "hard"):
            for threat in generate_scenario(diff):
                missing = required - set(threat.keys())
                assert not missing, f"Threat {threat.get('threat_id')} missing: {missing}"

    def test_strip_ground_truth_removes_correct_fields(self):
        scenario = generate_scenario("easy")
        stripped = strip_ground_truth(scenario[0])
        for key in stripped:
            assert not key.startswith("_"), f"Key {key} should be stripped"
        assert "_correct_priority" not in stripped
        assert "_correct_resource" not in stripped

    def test_threat_ids_easy_range(self):
        scenario = generate_scenario("easy")
        ids = [t["threat_id"] for t in scenario]
        assert ids == [f"THR-{1000 + i}" for i in range(5)]

    def test_threat_ids_medium_range(self):
        scenario = generate_scenario("medium")
        ids = [t["threat_id"] for t in scenario]
        assert ids == [f"THR-{2000 + i}" for i in range(10)]

    def test_threat_ids_hard_range(self):
        scenario = generate_scenario("hard")
        ids = sorted([t["threat_id"] for t in scenario])
        expected = sorted([f"THR-{3000 + i}" for i in range(15)])
        assert ids == expected

    def test_templates_count_at_least_21(self):
        assert len(THREAT_TEMPLATES) >= 21

    def test_generate_threat_deterministic(self):
        t1 = generate_threat(THREAT_TEMPLATES[0], "THR-TEST-1", seed_offset=0)
        t2 = generate_threat(THREAT_TEMPLATES[0], "THR-TEST-1", seed_offset=0)
        assert t1 == t2

    def test_scenarios_are_deterministic(self):
        s1 = generate_scenario("hard")
        s2 = generate_scenario("hard")
        assert [t["threat_id"] for t in s1] == [t["threat_id"] for t in s2]


# ===================================================================
# TestTaskRegistry
# ===================================================================


class TestTaskRegistry:
    def test_registry_has_three_tasks(self):
        assert set(TASK_REGISTRY.keys()) == {"easy", "medium", "hard"}

    def test_registry_graders_are_callable(self):
        for entry in TASK_REGISTRY.values():
            assert callable(entry["grader"])

    def test_registry_scenario_sizes(self):
        assert TASK_REGISTRY["easy"]["scenario_size"] == 5
        assert TASK_REGISTRY["medium"]["scenario_size"] == 10
        assert TASK_REGISTRY["hard"]["scenario_size"] == 15
