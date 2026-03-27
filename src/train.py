"""
train.py — Walk-forward training orchestrator with PARALLEL seed training.

For each window, all NUM_SEEDS agents are launched simultaneously as
independent subprocesses (one per seed).  Each subprocess runs
train_worker.py which:
  1. Trains a PPO agent for 7.5M timesteps using 10 SubprocVecEnv workers.
  2. Evaluates on the validation set.
  3. Saves the model and a small JSON result file.

The orchestrator waits for all seeds to finish, reads the JSON results,
picks the best (by mean DSR validation reward, per paper §5.2), and
passes its model path as the warm-start for the next window.

Speedup:  seeds run ~5× faster (parallelised) at the cost of ~5× more
CPU cores per window.  On a 96-core server this is ideal; on a 10-core
laptop, lower NUM_SEEDS or run sequentially with the original script.

Resource usage per window:
  NUM_SEEDS × (1 Python process + N_ENVS SubprocVecEnv workers)
  = 5 × 11 = 55 processes  (well within 96-core server capacity)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.MODELS_DIR, exist_ok=True)

WORKER = Path(__file__).parent / "train_worker.py"


# --------------------------------------------------------------------------- #
# Parallel walk-forward pipeline                                                #
# --------------------------------------------------------------------------- #

def train_pipeline():
    data_path   = config.DATA_DIR / "processed_features.csv"
    prices_path = config.DATA_DIR / "prices.csv"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run fetch_data.py first.")
        return
    if not prices_path.exists():
        print(f"ERROR: {prices_path} not found. Run fetch_data.py first.")
        return

    start_year      = int(config.START_DATE[:4])
    best_agent_path = None   # warm-start model from previous window

    for window in range(config.NUM_WINDOWS):
        train_start = f"{start_year + window}-01-01"
        train_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS - 1}-12-31"
        val_start   = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"

        print(f"\n{'='*60}")
        print(f"Window {window + 1}/{config.NUM_WINDOWS}  "
              f"[train {train_start[:4]}–{train_end[:4]}  |  val {val_start[:4]}]")
        print(f"Launching {config.NUM_SEEDS} seeds in parallel ...")
        print(f"{'='*60}")

        # ---------------------------------------------------------------- #
        # Launch all seeds simultaneously as independent subprocesses       #
        # ---------------------------------------------------------------- #
        procs = []
        for seed in range(config.NUM_SEEDS):
            cmd = [
                sys.executable, str(WORKER),
                "--window", str(window),
                "--seed",   str(seed),
            ]
            if best_agent_path and os.path.exists(str(best_agent_path)):
                cmd += ["--seed_model", str(best_agent_path)]

            # Inherit stdout/stderr so SB3 progress bars print to terminal
            proc = subprocess.Popen(cmd)
            procs.append((seed, proc))
            print(f"  [W{window} S{seed}] started (pid={proc.pid})")

        # ---------------------------------------------------------------- #
        # Wait for every seed to finish                                      #
        # ---------------------------------------------------------------- #
        print(f"\nWaiting for all {config.NUM_SEEDS} seeds to complete ...")
        for seed, proc in procs:
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"ERROR (code {proc.returncode})"
            print(f"  [W{window} S{seed}] finished — {status}")

        # ---------------------------------------------------------------- #
        # Collect results and select best seed (paper §5.2)                 #
        # ---------------------------------------------------------------- #
        best_val_reward        = -float("inf")
        best_window_agent_path = None

        print(f"\n  Results for window {window + 1}:")
        for seed in range(config.NUM_SEEDS):
            result_path = config.MODELS_DIR / f"result_window_{window}_seed_{seed}.json"
            if not result_path.exists():
                print(f"  [W{window} S{seed}] WARNING: no result file — skipping")
                continue
            with open(result_path) as f:
                r = json.load(f)
            print(f"    Seed {seed}: reward={r['val_reward']:+.6f}  return={r['val_return']:+.2%}")
            if r["val_reward"] > best_val_reward:
                best_val_reward        = r["val_reward"]
                best_window_agent_path = Path(r["model_path"])

        if best_window_agent_path is None:
            print(f"  WARNING: No valid seeds for window {window + 1}. Skipping warm-start.")
        else:
            print(f"\n  Best: {best_window_agent_path.name}  (reward={best_val_reward:+.6f})")

        # Warm-start next window from the best agent of this window
        best_agent_path = best_window_agent_path

    print("\nTraining complete.")


if __name__ == "__main__":
    train_pipeline()
