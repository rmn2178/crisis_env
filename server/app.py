"""FastAPI application for the Crisis Response Environment.

All endpoints match the OpenEnv specification exactly.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

# sys.path fix so imports resolve from /app
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import CrisisAction, CrisisObservation, CrisisStepResult
from server.environment import CrisisResponseEnvironment, VALID_TASK_IDS
from graders import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Crisis Response Environment",
    description="OpenEnv RL environment for AI Crisis Response & Rescue Coordination.",
    version="1.0.0",
)

env = CrisisResponseEnvironment()

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="easy | medium | hard")


class StepRequest(BaseModel):
    action_type: str
    threat_id: str
    resource_id: Optional[str] = None
    priority_level: Optional[str] = None
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """Health check endpoint — judges will hit this first."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "tasks_available": sorted(VALID_TASK_IDS),
    }


@app.get("/tasks", tags=["System"])
async def list_tasks() -> List[Dict[str, Any]]:
    """Return metadata for all available tasks."""
    tasks = []
    for task_id in ("easy", "medium", "hard"):
        entry = TASK_REGISTRY[task_id]
        tasks.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "difficulty": entry["difficulty"],
                "description": entry["description"],
                "scenario_size": entry["scenario_size"],
            }
        )
    return tasks


@app.post("/reset", response_model=CrisisObservation, tags=["OpenEnv"])
async def reset(body: ResetRequest) -> CrisisObservation:
    """Reset the environment to a new episode."""
    if body.task_id not in VALID_TASK_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{body.task_id}'. Must be one of: {sorted(VALID_TASK_IDS)}",
        )
    return env.reset(task_id=body.task_id)


@app.post("/step", response_model=CrisisStepResult, tags=["OpenEnv"])
async def step(body: StepRequest) -> CrisisStepResult:
    """Take one step in the environment."""
    action = CrisisAction(
        action_type=body.action_type,
        threat_id=body.threat_id,
        resource_id=body.resource_id,
        priority_level=body.priority_level,
        reasoning=body.reasoning,
    )
    return env.step(action)


@app.get("/state", tags=["OpenEnv"])
async def state() -> Dict[str, Any]:
    """Return current episode state."""
    return env.state.to_dict()
