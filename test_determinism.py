#!/usr/bin/env python3
"""
test_determinism.py — Verifies the environment produces identical 
outputs across 3 runs with the same seed.
"""
import subprocess
import sys
import os

SEED = 42
NUM_RUNS = 3


def run_episode(seed: int) -> dict:
    env = {"SEED": str(seed), "PYTHONPATH": "."}
    result = subprocess.run(
        ["python3", "inference.py"],
        capture_output=True, text=True,
        env={**os.environ.copy(), **env},
    )
    scores = {}
    for line in result.stdout.split("\n"):
        if "[SCORE]" in line:
            for part in line.split("|"):
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip().replace("[SCORE] ", "")
                    try:
                        scores[key] = float(val.strip())
                    except ValueError:
                        pass
    return scores


def main():
    print(f"Determinism test: seed={SEED}, {NUM_RUNS} runs\n")
    all_scores = [run_episode(SEED) for i in range(NUM_RUNS)
                  if print(f"Run {i+1}/{NUM_RUNS}...") or True]
    baseline = all_scores[0]
    all_match = all(
        all(abs(s.get(k, 0) - baseline.get(k, 0)) < 0.0001 for k in baseline)
        for s in all_scores[1:]
    )
    for i, s in enumerate(all_scores):
        print(f"Run {i+1}: final={s.get('final', 'N/A')}")
    print(f"\nDeterministic: {'PASS' if all_match else 'FAIL'}")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
