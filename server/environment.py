"""Core Crisis Response Environment logic.

exactly in structure: reset / step / state /
triaged, with deterministic scenario generation and dense per-step rewards.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import CrisisAction, CrisisObservation, CrisisStepResult
from scenario_generator import (
    ALL_RESOURCES,
    generate_scenario,
    strip_ground_truth,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TASK_IDS = {"easy", "medium", "hard"}
VALID_ACTION_TYPES = {"classify", "predict", "allocate", "rescue"}
VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

# Reward constants
REWARD_CORRECT_PRIORITY = 0.50
REWARD_CORRECT_RESOURCE = 0.30
REWARD_RESCUE_SPEED_BONUS = 0.20
PENALTY_WRONG_THREAT_ID = -0.10
PENALTY_INVALID_ACTION = -0.20
PENALTY_MISSED_CRITICAL = -0.20


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------


@dataclass
class EpisodeState:
    """Lightweight mutable state for one episode."""

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    task_id: str = "easy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CrisisResponseEnvironment:
    """OpenEnv-compliant RL environment for crisis response triage."""

    def __init__(self) -> None:
        self._state = EpisodeState()
        self._threats: List[Dict[str, Any]] = []
        self._pointer: int = 0
        self._triaged: List[Dict[str, Any]] = []
        self._cumulative_score: float = 0.0
        self._available_resources: List[str] = list(ALL_RESOURCES)
        self._done: bool = False

    # -- public properties ---------------------------------------------------

    @property
    def state(self) -> EpisodeState:
        return self._state

    @property
    def triaged(self) -> List[Dict[str, Any]]:
        return list(self._triaged)

    # -- reset ---------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> CrisisObservation:
        """Reset the environment for a new episode."""
        if task_id not in VALID_TASK_IDS:
            task_id = "easy"

        self._state = EpisodeState(task_id=task_id)
        self._threats = generate_scenario(task_id)
        self._pointer = 0
        self._triaged = []
        self._cumulative_score = 0.0
        self._available_resources = list(ALL_RESOURCES)
        self._done = False

        return self._build_observation(
            last_action_result="Episode started. Assess the first threat.",
            done=False,
        )

    # -- step ----------------------------------------------------------------

    def step(self, action: CrisisAction) -> CrisisStepResult:
        """Process one agent action and return the result."""
        if self._done:
            obs = self._build_observation("Episode already finished.", done=True)
            return CrisisStepResult(
                observation=obs, reward=0.0, done=True, info={"error": "episode_done"}
            )

        threat = self._current_threat()
        self._state.step_count += 1

        # 1. Validate action_type
        if action.action_type not in VALID_ACTION_TYPES:
            reward = max(0.0, min(1.0, PENALTY_INVALID_ACTION + 0.20))  # clip
            obs = self._build_observation(
                f"Invalid action_type '{action.action_type}'. Must be one of: {sorted(VALID_ACTION_TYPES)}",
                done=False,
            )
            return CrisisStepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"error": "invalid_action_type", "penalty": PENALTY_INVALID_ACTION},
            )

        # 2. Validate priority_level if provided
        if action.priority_level is not None and action.priority_level not in VALID_PRIORITIES:
            obs = self._build_observation(
                f"Invalid priority_level '{action.priority_level}'. Must be one of: {sorted(VALID_PRIORITIES)}",
                done=False,
            )
            return CrisisStepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"error": "invalid_priority_level", "penalty": PENALTY_INVALID_ACTION},
            )

        # 3. Validate threat_id matches current
        if action.threat_id != threat["threat_id"]:
            obs = self._build_observation(
                f"Wrong threat_id '{action.threat_id}'. Expected '{threat['threat_id']}'.",
                done=False,
            )
            return CrisisStepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"error": "wrong_threat_id", "penalty": PENALTY_WRONG_THREAT_ID},
            )

        # 4. Evaluate against ground truth
        reward, feedback = self._evaluate_action(action, threat)
        clipped_reward = round(max(0.0, min(1.0, reward)), 4)
        self._cumulative_score += clipped_reward

        # 5. Append to triaged
        self._triaged.append(
            {
                "threat_id": threat["threat_id"],
                "action": action.action_type,
                "priority_level": action.priority_level,
                "resource_id": action.resource_id,
                "reasoning": action.reasoning,
                "reward": clipped_reward,
                "correct_priority": threat["_correct_priority"],
                "correct_resource": threat["_correct_resource"],
                "step_number": self._state.step_count,
            }
        )

        # 5b. Remove used resource from available pool
        if action.resource_id and action.resource_id in self._available_resources:
            self._available_resources.remove(action.resource_id)

        # 6. Advance pointer, check done
        self._pointer += 1
        done = self._pointer >= len(self._threats)
        self._done = done

        obs = self._build_observation(feedback, done)
        return CrisisStepResult(
            observation=obs,
            reward=clipped_reward,
            done=done,
            info={
                "feedback": feedback,
                "step": self._state.step_count,
                "cumulative_score": round(self._cumulative_score, 4),
            },
        )

    # -- internal helpers ----------------------------------------------------

    def _current_threat(self) -> Dict[str, Any]:
        """Return the threat the agent should act on right now."""
        idx = min(self._pointer, len(self._threats) - 1)
        return self._threats[idx]

    def _evaluate_action(
        self, action: CrisisAction, threat: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Compare action against ground truth and compute reward + feedback."""
        reward = 0.0
        feedback_parts: List[str] = []

        correct_priority = threat["_correct_priority"]
        correct_resource = threat["_correct_resource"]

        # Priority check
        if action.priority_level == correct_priority:
            reward += REWARD_CORRECT_PRIORITY
            feedback_parts.append(f"Correct priority ({correct_priority}).")
        else:
            feedback_parts.append(
                f"Wrong priority: agent said {action.priority_level}, correct was {correct_priority}."
            )
            # CRITICAL miss penalty
            if correct_priority == "CRITICAL" and action.priority_level != "CRITICAL":
                reward += PENALTY_MISSED_CRITICAL
                feedback_parts.append("CRITICAL threat missed — penalty applied.")

        # Resource check (only relevant for allocate / rescue)
        if action.action_type in ("allocate", "rescue"):
            if action.resource_id == correct_resource:
                reward += REWARD_CORRECT_RESOURCE
                feedback_parts.append(f"Correct resource ({correct_resource}).")
            else:
                feedback_parts.append(
                    f"Wrong resource: agent used {action.resource_id}, optimal was {correct_resource}."
                )

        # Rescue speed bonus — if acted within first 3 steps
        if self._state.step_count <= 3:
            reward += REWARD_RESCUE_SPEED_BONUS
            feedback_parts.append("Speed bonus: responded within first 3 steps.")

        return reward, " ".join(feedback_parts)

    def _build_observation(
        self, last_action_result: str, done: bool
    ) -> CrisisObservation:
        """Construct the observation to send to the agent."""
        if done or self._pointer >= len(self._threats):
            # Terminal observation — use last threat data
            last_threat = self._threats[-1] if self._threats else {
                "threat_id": "NONE",
                "threat_type": "AIRSTRIKE",
                "location": "N/A",
                "severity": "LOW",
                "population_at_risk": 0,
                "time_to_impact": 0,
            }
            public = strip_ground_truth(last_threat)
            return CrisisObservation(
                threat_id=public.get("threat_id", "NONE"),
                threat_type=public.get("threat_type", "AIRSTRIKE"),
                location=public.get("location", "N/A"),
                severity=public.get("severity", "LOW"),
                population_at_risk=public.get("population_at_risk", 0),
                time_to_impact=public.get("time_to_impact", 0),
                available_resources=list(self._available_resources),
                threats_remaining=0,
                last_action_result=last_action_result,
                cumulative_score=round(self._cumulative_score, 4),
                done=True,
            )

        threat = self._current_threat()
        public = strip_ground_truth(threat)
        threats_remaining = len(self._threats) - self._pointer

        return CrisisObservation(
            threat_id=public["threat_id"],
            threat_type=public["threat_type"],
            location=public["location"],
            severity=public["severity"],
            population_at_risk=public["population_at_risk"],
            time_to_impact=public["time_to_impact"],
            available_resources=list(self._available_resources),
            threats_remaining=threats_remaining,
            last_action_result=last_action_result,
            cumulative_score=round(self._cumulative_score, 4),
            done=False,
        )
