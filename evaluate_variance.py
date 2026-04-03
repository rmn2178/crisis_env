#!/usr/bin/env python3
"""
evaluate_variance.py — Evaluates score variance across multiple episodes.
Runs 10 episodes with the same seed and computes mean/std deviation.
"""
import subprocess
import sys
import statistics

SEED = 42
NUM_EPISODES = 10


def run_episode(seed: int) -> float:
    env = {"SEED": str(seed), "PYTHONPATH": "."}
    result = subprocess.run(
        ["python3", "inference.py"],
        capture_output=True, text=True,
        env={**__import__("os").environ.copy(), **env},
    )
    for line in result.stdout.split("\n"):
        if "final=" in line and "[SCORE]" in line:
            for part in line.split("|"):
                if "final=" in part:
                    try:
                        return float(part.split("=")[1].strip())
                    except ValueError:
                        pass
    return 0.0


def main():
    print(f"Running variance evaluation: seed={SEED}, {NUM_EPISODES} episodes\n")
    scores = []
    for i in range(NUM_EPISODES):
        print(f"Episode {i+1}/{NUM_EPISODES}...", end=" ", flush=True)
        score = run_episode(SEED)
        scores.append(score)
        print(f"Score: {score:.4f}")
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"\nMean: {mean_score:.4f} | Std: {std_dev:.4f} | "
          f"Min: {min(scores):.4f} | Max: {max(scores):.4f}")
    label = "LOW" if std_dev < 0.01 else "ACCEPTABLE" if std_dev < 0.05 else "HIGH"
    print(f"Variance: {label}")
    sys.exit(0)


if __name__ == "__main__":
    main()
