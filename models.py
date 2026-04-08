"""Pydantic v2 models for the Crisis Response Environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CrisisAction(BaseModel):
    """What the agent sends to /step."""

    action_type: str = Field(
        ...,
        description="One of: classify, predict, allocate, rescue",
    )
    threat_id: str = Field(
        ...,
        description="ID of the threat being acted on, e.g. THR-001",
    )
    resource_id: Optional[str] = Field(
        default=None,
        description="e.g. RES-fighter_jet-01 — required for allocate/rescue actions",
    )
    priority_level: Optional[str] = Field(
        default=None,
        description="CRITICAL | HIGH | MEDIUM | LOW",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="One-sentence explanation — not scored, used for logging only",
    )


class CrisisObservation(BaseModel):
    """What /reset and the observation field of /step return."""

    threat_id: str = Field(..., description="Current threat to act on")
    threat_type: str = Field(
        ...,
        description="AIRSTRIKE | SHIP_ATTACK | DRONE_THREAT",
    )
    location: str = Field(..., description="e.g. Military Base Alpha")
    severity: str = Field(..., description="CRITICAL | HIGH | MEDIUM | LOW")
    population_at_risk: int = Field(..., description="Number of lives at risk")
    time_to_impact: int = Field(
        ..., description="Seconds until event escalates"
    )
    available_resources: List[str] = Field(
        ..., description="List of resource IDs still available"
    )
    threats_remaining: int = Field(
        ..., description="Threats left in this episode"
    )
    last_action_result: str = Field(
        ..., description="Feedback from previous step"
    )
    cumulative_score: float = Field(..., description="Running reward total")
    done: bool = Field(..., description="True when all threats handled")


class CrisisStepResult(BaseModel):
    """What /step returns (OpenEnv spec)."""

    observation: CrisisObservation
    reward: float = Field(..., description="Per-step reward clipped to [0.0, 1.0]")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
