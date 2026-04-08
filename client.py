"""HTTP client for interacting with the Crisis Response Environment API.

Provides a clean Python interface for programmatic access to the
running server (useful for testing, scripting, and remote inference).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from models import CrisisAction, CrisisObservation, CrisisStepResult


class CrisisResponseClient:
    """Synchronous HTTP client for the Crisis Response Environment."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # -- System endpoints ----------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """GET /health"""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list:
        """GET /tasks"""
        resp = self.session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """GET /state"""
        resp = self.session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    # -- OpenEnv endpoints ---------------------------------------------------

    def reset(self, task_id: str = "easy") -> CrisisObservation:
        """POST /reset"""
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return CrisisObservation(**resp.json())

    def step(
        self,
        action_type: str,
        threat_id: str,
        resource_id: Optional[str] = None,
        priority_level: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> CrisisStepResult:
        """POST /step"""
        body: Dict[str, Any] = {
            "action_type": action_type,
            "threat_id": threat_id,
        }
        if resource_id is not None:
            body["resource_id"] = resource_id
        if priority_level is not None:
            body["priority_level"] = priority_level
        if reasoning is not None:
            body["reasoning"] = reasoning

        resp = self.session.post(f"{self.base_url}/step", json=body)
        resp.raise_for_status()
        return CrisisStepResult(**resp.json())

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
