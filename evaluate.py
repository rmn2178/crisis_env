"""
evaluate.py — Compare baseline (rule-based) vs trained RL agent.

Runs both agents over the same seeds and prints a side-by-side comparison.

Usage:
    python3 evaluate.py                          # default 10 eval episodes
    python3 evaluate.py --episodes 20            # more episodes
    python3 evaluate.py --checkpoint checkpoints/model.pt  # specific checkpoint
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from policy_model import PolicyNetwork
from utils import (
    EpisodeSummary,
    build_state_vector,
    build_action_candidates,
    candidate_tensor,
    observation_to_dict,
    state_to_metrics,
    load_checkpoint,
    set_global_seed,
)
from models import CrisisAction
from server.environment import CrisisEnvironment


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_EPISODES   = 10
DEFAULT_CHECKPOINT = Path("checkpoints/model.pt")
STATE_DIM  = 229
ACTION_DIM = 45


# ─────────────────────────────────────────────
# BASELINE AGENT (rule-based, same as inference.py)
# ─────────────────────────────────────────────

def run_baseline_episode(seed: int) -> EpisodeSummary:
    """Run the deterministic baseline agent locally."""
    env = CrisisEnvironment(seed=seed)
    observation = observation_to_dict(env.reset())

    threats = observation.get("threats", [])
    resources = observation.get("resources", [])
    zones = observation.get("affected_zones", [])
    done = False
    total_reward = 0.0

    classified = set()
    predicted = set()
    evacuated = set()
    allocated = set()
    coordinated = False

    def priority_score(t):
        return (t["severity"] * t["population_at_risk"]) / max(t["time_to_impact"], 1)

    while not done:
        actions = []
        active = [t for t in threats if t.get("status") == "active"]
        impacted = [z for z in zones if z.get("is_active", False)]

        # Classify
        for t in active:
            if t["threat_id"] not in classified:
                actions.append({
                    "action_type": "classify",
                    "classification": {
                        "threat_id": t["threat_id"],
                        "predicted_type": t["threat_type"],
                        "predicted_severity": t["severity"],
                    },
                })
                classified.add(t["threat_id"])

        # Predict
        for t in active:
            if t["threat_id"] not in predicted:
                tti = max(int(t.get("time_to_impact", 5)), 1)
                pop = int(t.get("population_at_risk", 100))
                sev = float(t.get("severity", 5.0))
                est_pop = int(pop * (0.8 + sev / 50.0))
                actions.append({
                    "action_type": "predict",
                    "prediction": {
                        "threat_id": t["threat_id"],
                        "predicted_tti": tti,
                        "predicted_pop": est_pop,
                    },
                })
                predicted.add(t["threat_id"])

        # Evacuate
        for t in active:
            if t["threat_id"] not in evacuated:
                tti = t.get("time_to_impact", 0)
                pop = t.get("population_at_risk", 0)
                if tti > 2 and pop > 0:
                    units = min(5, max(2, int(pop / 100)))
                    actions.append({
                        "action_type": "evacuate",
                        "evacuate": {
                            "threat_id": t["threat_id"],
                            "evac_units": units,
                        },
                    })

        # Coordinate
        if active and not coordinated:
            ranked = sorted(active, key=priority_score, reverse=True)
            actions.append({
                "action_type": "coordinate",
                "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
            })
            coordinated = True

        # Allocate
        ranked = sorted(active, key=priority_score, reverse=True)
        avail = [r for r in resources if r.get("is_available", False)]
        for t in ranked:
            if t["threat_id"] not in allocated and t.get("assigned_resource") is None and avail:
                best = max(avail, key=lambda r: r["effectiveness"])
                actions.append({
                    "action_type": "allocate",
                    "allocation": {"threat_id": t["threat_id"], "resource_id": best["resource_id"]},
                })
                allocated.add(t["threat_id"])
                avail.remove(best)

        # Rescue
        for z in impacted:
            remaining = z.get("total_victims", 0) - z.get("rescued", 0)
            if remaining > 0:
                actions.append({
                    "action_type": "rescue",
                    "rescue": {"zone_id": z["zone_id"], "rescue_units_to_send": 5},
                })

        # Execute first action
        if actions:
            action = actions[0]
        elif active:
            ranked = sorted(active, key=priority_score, reverse=True)
            action = {
                "action_type": "coordinate",
                "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
            }
        else:
            break

        result = env.step(CrisisAction(**action))
        total_reward += float(result.reward)
        done = result.done
        observation = observation_to_dict(result.observation)
        threats = observation.get("threats", threats)
        resources = observation.get("resources", resources)
        zones = observation.get("affected_zones", zones)

        # Track executed evacuate action
        if action.get("action_type") == "evacuate":
            evacuated.add(action["evacuate"]["threat_id"])

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=state.step_count,
    )


# ─────────────────────────────────────────────
# TRAINED AGENT
# ─────────────────────────────────────────────

def run_trained_episode(
    env: CrisisEnvironment,
    policy: PolicyNetwork,
    device: str = "cpu",
) -> EpisodeSummary:
    """Run one episode with the trained policy (greedy — no sampling)."""
    observation = env.reset()
    obs_dict = observation_to_dict(observation)
    total_reward = 0.0
    done = False

    while not done:
        state_vec = build_state_vector(obs_dict)
        candidates = build_action_candidates(obs_dict)

        if not candidates:
            active = [t for t in obs_dict.get("threats", []) if t.get("status") == "active"]
            if active:
                priority = sorted(
                    active,
                    key=lambda t: (t["severity"] * t["population_at_risk"]) / max(t["time_to_impact"], 1),
                    reverse=True,
                )
                action = CrisisAction(**{
                    "action_type": "coordinate",
                    "coordination": {"priority_order": [t["threat_id"] for t in priority]},
                })
            else:
                action = CrisisAction(**{
                    "action_type": "coordinate",
                    "coordination": {"priority_order": []},
                })
            result = env.step(action)
            total_reward += float(result.reward)
            done = result.done
            obs_dict = observation_to_dict(result.observation)
            continue

        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device)
        action_tensor = candidate_tensor(candidates).to(device)

        with torch.no_grad():
            logits = policy(state_tensor, action_tensor)

        # Greedy selection (argmax) during evaluation
        action_idx = torch.argmax(logits).item()

        selected = candidates[action_idx]
        crisis_action = CrisisAction(**selected.action)

        result = env.step(crisis_action)
        total_reward += float(result.reward)
        done = result.done
        obs_dict = observation_to_dict(result.observation)

    state = env.state()
    task_scores = state_to_metrics(state)
    return EpisodeSummary(
        total_reward=round(total_reward, 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=state.step_count,
    )


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    num_episodes: int = DEFAULT_EPISODES,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Run evaluation comparing baseline vs trained agent."""

    # Load trained policy
    policy = PolicyNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    metadata = load_checkpoint(checkpoint_path, policy, device=device)
    policy.eval()

    print(f"{'='*60}")
    print(f"  Evaluation: Baseline vs Trained Agent")
    print(f"{'='*60}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Model meta:  ep={metadata.get('episode', '?')}, "
          f"best_score={metadata.get('best_score', '?')}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    baseline_results: List[EpisodeSummary] = []
    trained_results: List[EpisodeSummary] = []

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep

        # Baseline
        baseline_summary = run_baseline_episode(ep_seed)
        baseline_results.append(baseline_summary)

        # Trained
        env = CrisisEnvironment(seed=ep_seed)
        trained_summary = run_trained_episode(env, policy, device)
        trained_results.append(trained_summary)

        print(
            f"Episode {ep:2d} (seed={ep_seed}) | "
            f"Baseline: reward={baseline_summary.total_reward:7.2f} "
            f"score={baseline_summary.final_score:.4f} | "
            f"Trained: reward={trained_summary.total_reward:7.2f} "
            f"score={trained_summary.final_score:.4f} | "
            f"{'▲' if trained_summary.final_score > baseline_summary.final_score else '▼'}"
        )

    # Aggregate statistics
    def avg_field(results: List[EpisodeSummary], field: str) -> float:
        return sum(getattr(r, field) for r in results) / max(len(results), 1)

    def avg_task_score(results: List[EpisodeSummary], task: str) -> float:
        return sum(r.task_scores.get(task, 0.0) for r in results) / max(len(results), 1)

    print(f"\n{'='*60}")
    print(f"  RESULTS (averaged over {num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Trained':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    b_reward = avg_field(baseline_results, "total_reward")
    t_reward = avg_field(trained_results, "total_reward")
    print(f"  {'Total Reward':<25} {b_reward:>12.2f} {t_reward:>12.2f} {t_reward-b_reward:>+10.2f}")

    b_score = avg_field(baseline_results, "final_score")
    t_score = avg_field(trained_results, "final_score")
    print(f"  {'Final Score':<25} {b_score:>12.4f} {t_score:>12.4f} {t_score-b_score:>+10.4f}")

    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for task in ["classification", "prediction", "allocation", "coordination", "rescue", "evacuation"]:
        b = avg_task_score(baseline_results, task)
        t = avg_task_score(trained_results, task)
        print(f"  {task.title():<25} {b:>12.4f} {t:>12.4f} {t-b:>+10.4f}")

    print(f"{'='*60}")

    wins = sum(1 for b, t in zip(baseline_results, trained_results) if t.final_score > b.final_score)
    print(f"\n  Trained won: {wins}/{num_episodes} episodes ({100*wins/num_episodes:.0f}%)")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent vs baseline")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of evaluation episodes")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, mps, auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    set_global_seed(args.seed)

    evaluate(
        num_episodes=args.episodes,
        checkpoint_path=Path(args.checkpoint),
        seed=args.seed,
        device=device,
    )
