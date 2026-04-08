"""OpenEnv wrapper for the Crisis Response Environment.

Mirrors bug_triage_env_environment.py exactly — wraps the core environment
class and exposes the OpenEnv ``Environment`` interface.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import CrisisAction, CrisisObservation, CrisisStepResult
from server.environment import CrisisResponseEnvironment

# Try importing the OpenEnv base class; fall back gracefully if missing.
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    Environment = object  # type: ignore[assignment,misc]
    State = None  # type: ignore[assignment,misc]


class CrisisResponseEnvEnvironment(Environment):  # type: ignore[misc]
    """OpenEnv-compliant wrapper around :class:`CrisisResponseEnvironment`.

    This is the class the OpenEnv harness instantiates.  It delegates all
    logic to the core environment and translates between OpenEnv types and
    our own Pydantic models.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        self._env = CrisisResponseEnvironment()
        if State is not None:
            self._state = State(
                episode_id=self._env.state.episode_id,
                step_count=self._env.state.step_count,
            )
        else:
            self._state = None

    # -- OpenEnv interface ---------------------------------------------------

    def reset(self, task_id: str = "easy") -> CrisisObservation:
        """Reset environment for a new episode and return the first observation."""
        obs = self._env.reset(task_id=task_id)
        self._sync_state()
        return obs

    def step(self, action: CrisisAction) -> CrisisStepResult:
        """Process one agent action."""
        result = self._env.step(action)
        self._sync_state()
        return result

    # -- Properties ----------------------------------------------------------

    @property
    def state(self):
        """Return the current OpenEnv State (or EpisodeState fallback)."""
        if self._state is not None:
            return self._state
        return self._env.state

    @property
    def triaged(self) -> List[Dict[str, Any]]:
        """Return the list of triaged action records."""
        return self._env.triaged

    # -- Internals -----------------------------------------------------------

    def _sync_state(self) -> None:
        """Keep the OpenEnv State object in sync with the core environment."""
        if State is not None and self._state is not None:
            self._state.episode_id = self._env.state.episode_id
            self._state.step_count = self._env.state.step_count
