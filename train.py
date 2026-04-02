"""
train.py — REINFORCE training loop for the Crisis Response Environment.

Trains a policy network locally against CrisisEnvironment (no HTTP server needed).
Uses candidate-action scoring: the policy scores each valid action and samples
proportionally via softmax, then updates with policy gradient + entropy bonus.

Usage:
    python3 train.py                          # defaults: 500 episodes, gamma=0.99
    python3 train.py --episodes 1000          # longer training
    python3 train.py --lr 1e-3 --gamma 0.95   # custom hyperparameters
    python3 train.py --seed 123               # different seed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from policy_model import PolicyNetwork
from utils import (
    CandidateAction,
    EpisodeSummary,
    TrainingLogger,
    build_state_vector,
    build_action_candidates,
    candidate_tensor,
    compute_discounted_returns,
    moving_average,
    save_checkpoint,
    load_checkpoint,
    observation_to_dict,
    state_to_metrics,
    set_global_seed,
)
from models import CrisisAction
from server.environment import CrisisEnvironment


# ─────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_EPISODES  = 500
DEFAULT_LR        = 3e-4
DEFAULT_GAMMA     = 0.99
DEFAULT_HIDDEN    = 256
DEFAULT_SEED      = 42
DEFAULT_ENTROPY   = 0.01   # entropy bonus coefficient (exploration)
DEFAULT_LOG_WINDOW = 20    # moving average window
CHECKPOINT_DIR    = Path("checkpoints")
LOG_DIR           = Path("logs")

# Feature dimensions (must match utils.py)
STATE_DIM  = 229  # build_state_vector output size
ACTION_DIM = 45   # candidate_features output size (44 base + 1 optional score/pad)


# ─────────────────────────────────────────────
# SINGLE EPISODE ROLLOUT (REINFORCE)
# ─────────────────────────────────────────────

def rollout_episode(
    env: CrisisEnvironment,
    policy: PolicyNetwork,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], EpisodeSummary]:
    """
    Run one full episode using the policy network.
    Returns (log_probs, entropies, rewards, summary).
    """
    observation = env.reset()
    obs_dict = observation_to_dict(observation)

    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    done = False

    while not done:
        # Build state vector and candidate actions
        state_vec = build_state_vector(obs_dict)
        candidates = build_action_candidates(obs_dict)

        if not candidates:
            # No valid candidates — send coordinate as fallback
            active = [t for t in obs_dict.get("threats", []) if t.get("status") == "active"]
            if active:
                priority = sorted(
                    active,
                    key=lambda t: (t["severity"] * t["population_at_risk"]) / max(t["time_to_impact"], 1),
                    reverse=True,
                )
                fallback_action = CrisisAction(**{
                    "action_type": "coordinate",
                    "coordination": {"priority_order": [t["threat_id"] for t in priority]},
                })
            else:
                fallback_action = CrisisAction(**{
                    "action_type": "coordinate",
                    "coordination": {"priority_order": []},
                })
            result = env.step(fallback_action)
            rewards.append(float(result.reward))
            done = result.done
            obs_dict = observation_to_dict(result.observation)
            continue

        # Forward pass through policy
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device)
        action_tensor = candidate_tensor(candidates).to(device)

        logits = policy(state_tensor, action_tensor)

        # Sample action from categorical distribution
        dist = Categorical(logits=logits)
        action_idx = dist.sample()

        log_probs.append(dist.log_prob(action_idx))
        entropies.append(dist.entropy())

        # Execute selected action
        selected = candidates[action_idx.item()]
        crisis_action = CrisisAction(**selected.action)

        result = env.step(crisis_action)
        rewards.append(float(result.reward))
        done = result.done
        obs_dict = observation_to_dict(result.observation)

    # Build episode summary
    state = env.state()
    task_scores = state_to_metrics(state)
    summary = EpisodeSummary(
        total_reward=round(sum(rewards), 4),
        final_score=round(task_scores["final"], 4),
        task_scores=task_scores,
        steps=state.step_count,
    )

    return log_probs, entropies, rewards, summary


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    num_episodes: int = DEFAULT_EPISODES,
    lr: float = DEFAULT_LR,
    gamma: float = DEFAULT_GAMMA,
    hidden_dim: int = DEFAULT_HIDDEN,
    entropy_coeff: float = DEFAULT_ENTROPY,
    seed: int = DEFAULT_SEED,
    checkpoint_every: int = 50,
    log_window: int = DEFAULT_LOG_WINDOW,
    device: str = "cpu",
) -> None:
    """Main training loop."""

    set_global_seed(seed)

    # Initialise policy
    policy = PolicyNetwork(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Logging
    logger = TrainingLogger(LOG_DIR / "training.jsonl")

    print(f"{'='*60}")
    print(f"  Crisis Response RL Training (REINFORCE)")
    print(f"{'='*60}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Learning rate:  {lr}")
    print(f"  Gamma:          {gamma}")
    print(f"  Hidden dim:     {hidden_dim}")
    print(f"  Entropy coeff:  {entropy_coeff}")
    print(f"  Seed:           {seed}")
    print(f"  Device:         {device}")
    print(f"{'='*60}\n")

    reward_history: List[float] = []
    score_history: List[float] = []
    best_score = 0.0
    avg_reward = 0.0
    avg_score = 0.0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Vary seed per episode for diverse experience
        env = CrisisEnvironment(seed=seed + episode)

        log_probs, entropies, rewards, summary = rollout_episode(env, policy, device)

        if not log_probs:
            continue

        # Compute discounted returns
        returns = compute_discounted_returns(rewards, gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise returns (reduces variance)
        if returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Policy gradient loss (REINFORCE)
        policy_loss = torch.stack([
            -log_prob * ret
            for log_prob, ret in zip(log_probs, returns_tensor)
        ]).sum()

        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy_coeff * torch.stack(entropies).sum()

        loss = policy_loss + entropy_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        reward_history.append(summary.total_reward)
        score_history.append(summary.final_score)

        avg_reward = moving_average(reward_history, log_window)
        avg_score = moving_average(score_history, log_window)

        # Log
        logger.write({
            "episode": episode,
            "total_reward": summary.total_reward,
            "final_score": summary.final_score,
            "task_scores": summary.task_scores,
            "steps": summary.steps,
            "avg_reward": round(avg_reward, 4),
            "avg_score": round(avg_score, 4),
        })

        # Print progress
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            eps_per_sec = episode / max(elapsed, 1e-6)
            print(
                f"Episode {episode:4d} | "
                f"reward: {summary.total_reward:7.2f} | "
                f"score: {summary.final_score:.4f} | "
                f"avg_reward: {avg_reward:7.2f} | "
                f"avg_score: {avg_score:.4f} | "
                f"steps: {summary.steps:2d} | "
                f"{eps_per_sec:.1f} ep/s"
            )

        # Save best model
        if summary.final_score > best_score:
            best_score = summary.final_score
            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pt",
                policy, optimizer,
                metadata={
                    "episode": episode,
                    "final_score": best_score,
                    "seed": seed,
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "hidden_dim": hidden_dim,
                },
            )

        # Periodic checkpoint
        if episode % checkpoint_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / f"checkpoint_ep{episode}.pt",
                policy, optimizer,
                metadata={
                    "episode": episode,
                    "final_score": summary.final_score,
                    "avg_score": avg_score,
                    "seed": seed,
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "hidden_dim": hidden_dim,
                },
            )

    # Final save
    save_checkpoint(
        CHECKPOINT_DIR / "model.pt",
        policy, optimizer,
        metadata={
            "episode": num_episodes,
            "final_score": score_history[-1] if score_history else 0.0,
            "best_score": best_score,
            "seed": seed,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": hidden_dim,
        },
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best final score: {best_score:.4f}")
    print(f"  Last avg score (last {log_window}): {avg_score:.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}/")
    print(f"  Logs saved to: {LOG_DIR}/training.jsonl")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for Crisis Response Environment")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="Hidden layer dimension")
    parser.add_argument("--entropy", type=float, default=DEFAULT_ENTROPY, help="Entropy bonus coefficient")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, mps, auto")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    train(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        entropy_coeff=args.entropy,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        device=device,
    )
