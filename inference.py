"""Inference script for the Crisis Response Environment.

MUST be in project root (not server/).
Uses OpenAI client, reads credentials from env vars, runs all 3 tasks,
prints scores in the exact log format judges parse.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI

from models import CrisisAction, CrisisObservation, CrisisStepResult
from server.crisis_response_env_environment import CrisisResponseEnvEnvironment
from graders import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert military crisis coordinator triaging simultaneous emergency events.

For each threat you receive, return a JSON object with your response decision.
Return ONLY the JSON object — no preamble, no explanation, no markdown fences.

Your decision must follow this exact schema:
{
  "action_type":    "<type>",
  "threat_id":      "<exact threat_id from the observation>",
  "priority_level": "<priority>",
  "resource_id":    "<resource_id if allocating/rescuing, otherwise null>",
  "reasoning":      "<one sentence explaining your decision>"
}

action_type must be exactly one of:
  classify  — identify the threat type and priority level
  predict   — forecast time to impact and population affected
  allocate  — assign the best available resource unit to intercept
  rescue    — deploy rescue unit after impact for casualty extraction

priority_level must be exactly one of:
  CRITICAL — mass casualty event imminent, all personnel at risk
  HIGH     — major threat, significant casualties without response
  MEDIUM   — threat contained but needs monitoring, limited risk
  LOW      — low risk, standard procedure sufficient

Rules for resource allocation:
  - fighter_jet: use ONLY for AIRSTRIKE threats
  - naval_vessel: use ONLY for SHIP_ATTACK threats
  - ground_unit: use for AIRSTRIKE or DRONE_THREAT
  - medic_team: use ONLY for rescue actions after impact
  - evacuation_unit: use for DRONE_THREAT or SHIP_ATTACK evacuation

Rules for prioritization:
  - Always handle CRITICAL threats before HIGH
  - CRITICAL with lower time_to_impact = highest urgency
  - Never skip a CRITICAL threat to handle a MEDIUM

Strong signals:
  - "smoke visible", "explosion reported" = CRITICAL AIRSTRIKE
  - "vessel taking on water", "SOS signal" = CRITICAL SHIP_ATTACK
  - "drone spotted", "unauthorized UAV" = HIGH or CRITICAL DRONE_THREAT
  - time_to_impact < 60 = CRITICAL regardless of type
  - population_at_risk > 500 = always CRITICAL or HIGH
"""

# ---------------------------------------------------------------------------
# Logging helpers (exact format — judges parse stdout)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_user_prompt(obs: Dict[str, Any], seen_threats: List[Dict[str, Any]]) -> str:
    """Build a user prompt from the current observation.

    Shows last 10 seen threats as context for cascade detection, plus all
    fields of the current threat.
    """
    parts: List[str] = []

    # Context from previous threats
    if seen_threats:
        recent = seen_threats[-10:]
        parts.append("=== Previous threats you have handled ===")
        for i, t in enumerate(recent, 1):
            parts.append(
                f"  {i}. {t.get('threat_id','?')} | type={t.get('threat_type','?')} "
                f"| severity={t.get('severity','?')} | location={t.get('location','?')} "
                f"| result={t.get('last_action_result','?')}"
            )
        parts.append("")

    # Current threat
    parts.append("=== Current threat requiring your decision ===")
    parts.append(f"  threat_id:          {obs.get('threat_id', '?')}")
    parts.append(f"  threat_type:        {obs.get('threat_type', '?')}")
    parts.append(f"  location:           {obs.get('location', '?')}")
    parts.append(f"  severity:           {obs.get('severity', '?')}")
    parts.append(f"  population_at_risk: {obs.get('population_at_risk', '?')}")
    parts.append(f"  time_to_impact:     {obs.get('time_to_impact', '?')}s")
    parts.append(f"  available_resources: {obs.get('available_resources', [])}")
    parts.append(f"  threats_remaining:  {obs.get('threats_remaining', '?')}")
    parts.append(f"  last_action_result: {obs.get('last_action_result', '?')}")
    parts.append("")
    parts.append("Return your crisis response decision as a JSON object now.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(
    client: OpenAI,
    user_prompt: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Call the LLM and parse a JSON action dict from the response.

    Strips markdown fences, validates required fields, retries on failure.
    Returns ``None`` on total failure.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""

            # Strip markdown fences
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            # Validate required fields
            if "action_type" not in parsed or "threat_id" not in parsed:
                raise ValueError("Missing required fields: action_type, threat_id")

            return parsed

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[WARN] LLM call failed after {retries} attempts: {exc}", flush=True)
                return None

    return None


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------


def fallback_action(threat_id: str) -> Dict[str, Any]:
    """Safe fallback that never causes a penalty."""
    return {
        "action_type": "classify",
        "threat_id": threat_id,
        "priority_level": "HIGH",
        "resource_id": None,
        "reasoning": "Fallback: defaulting to classification",
    }


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


async def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task end-to-end and return the graded score."""
    log_start(task_id, ENV_URL, MODEL_NAME)

    env = CrisisResponseEnvEnvironment()
    obs = env.reset(task_id=task_id)

    rewards: List[float] = []
    seen_threats: List[Dict[str, Any]] = []
    step_num = 0
    done = False

    while not done:
        step_num += 1
        obs_dict = obs.model_dump()

        # Build prompt
        user_prompt = build_user_prompt(obs_dict, seen_threats)

        # Call LLM
        action_dict = call_llm(client, user_prompt)
        if action_dict is None:
            action_dict = fallback_action(obs_dict["threat_id"])

        # Ensure threat_id matches current observation
        action_dict["threat_id"] = obs_dict["threat_id"]

        # Step
        try:
            crisis_action = CrisisAction(**action_dict)
            result = env.step(crisis_action)
            reward = result.reward
            done = result.done
            obs = result.observation
            error = result.info.get("error")
        except Exception as exc:
            reward = 0.0
            done = False
            error = str(exc)
            # Try to recover observation
            try:
                obs = env.reset(task_id=task_id)
            except Exception:
                break

        rewards.append(reward)

        # Log step
        log_step(step_num, action_dict, reward, done, error)

        # Track seen threats for context
        seen_threats.append(
            {
                "threat_id": obs_dict.get("threat_id"),
                "threat_type": obs_dict.get("threat_type"),
                "severity": obs_dict.get("severity"),
                "location": obs_dict.get("location"),
                "last_action_result": obs_dict.get("last_action_result"),
            }
        )

        # Safety: prevent infinite loops
        if step_num > 100:
            break

    # Compute grader score
    grader = TASK_REGISTRY[task_id]["grader"]
    triaged = env.triaged
    score = grader(triaged)

    log_end(success=True, steps=step_num, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all 3 tasks, print summary table with average."""
    start_time = time.time()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    task_ids = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for task_id in task_ids:
        score = await run_task(client, task_id)
        scores[task_id] = score

    elapsed = time.time() - start_time

    # Summary table
    print("\n" + "=" * 60)
    print(f"  {'task':<12} {'score':>8}")
    print("-" * 24)
    for task_id in task_ids:
        s = scores[task_id]
        bar = "\u2588" * int(s * 30)
        print(f"  {task_id:<12} {s:>8.3f}  {bar}")
    print("-" * 24)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<12} {avg:>8.3f}")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
