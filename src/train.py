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
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs(config.MODELS_DIR, exist_ok=True)

WORKER = Path(__file__).parent / "train_worker.py"


# --------------------------------------------------------------------------- #
# Parallel walk-forward pipeline                                                #
# --------------------------------------------------------------------------- #

def train_pipeline():
    data_path   = config.DATA_DIR / "processed_features.csv"
    prices_path = config.DATA_DIR / "prices.csv"

    if not data_path.exists():
        logger.error("%s not found. Run fetch_data.py first.", data_path)
        return
    if not prices_path.exists():
        logger.error("%s not found. Run fetch_data.py first.", prices_path)
        return

    start_year      = int(config.START_DATE[:4])
    best_agent_path = None   # warm-start model from previous window

    for window in range(config.NUM_WINDOWS):
        train_start = f"{start_year + window}-01-01"
        train_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS - 1}-12-31"
        val_start   = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"

        logger.info("=" * 60)
        logger.info(
            "Window %d/%d  [train %s–%s  |  val %s]",
            window + 1, config.NUM_WINDOWS,
            train_start[:4], train_end[:4], val_start[:4],
        )
        logger.info("Launching %d seeds in parallel ...", config.NUM_SEEDS)
        logger.info("=" * 60)

        # ---------------------------------------------------------------- #
        # Launch all seeds simultaneously as independent subprocesses       #
        # ---------------------------------------------------------------- #
        # Limit each subprocess to 1 OMP/MKL thread to prevent oversubscription.
        # With 5 seeds × 11 processes (1 main + 10 SubprocVecEnv workers) = 55
        # processes, uncapped PyTorch OMP threads (default ~8-10 each) would
        # spawn 500+ threads on 96 cores, killing performance.
        worker_env = os.environ.copy()
        worker_env.update({
            "OMP_NUM_THREADS":        "1",
            "MKL_NUM_THREADS":        "1",
            "OPENBLAS_NUM_THREADS":   "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS":    "1",
        })

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
            proc = subprocess.Popen(cmd, env=worker_env)
            procs.append((seed, proc))
            logger.info("[W%d S%d] started (pid=%d)", window, seed, proc.pid)

        # ---------------------------------------------------------------- #
        # Wait for every seed to finish                                      #
        # ---------------------------------------------------------------- #
        logger.info("Waiting for all %d seeds to complete ...", config.NUM_SEEDS)
        for seed, proc in procs:
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"ERROR (code {proc.returncode})"
            logger.info("[W%d S%d] finished — %s", window, seed, status)

        # ---------------------------------------------------------------- #
        # Collect results and select best seed (paper §5.2)                 #
        # ---------------------------------------------------------------- #
        best_val_reward        = -float("inf")
        best_window_agent_path = None

        logger.info("Results for window %d:", window + 1)
        for seed in range(config.NUM_SEEDS):
            result_path = config.MODELS_DIR / f"result_window_{window}_seed_{seed}.json"
            if not result_path.exists():
                logger.warning("[W%d S%d] no result file — skipping", window, seed)
                continue
            try:
                with open(result_path) as f:
                    r = json.load(f)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("[W%d S%d] corrupt result file (%s) — skipping", window, seed, exc)
                continue
            logger.info(
                "  Seed %d: reward=%+.6f  return=%+.2f%%",
                seed, r["val_reward"], r["val_return"] * 100,
            )
            if r["val_reward"] > best_val_reward:
                best_val_reward        = r["val_reward"]
                best_window_agent_path = Path(r["model_path"])

        if best_window_agent_path is None:
            logger.warning("No valid seeds for window %d. Skipping warm-start.", window + 1)
        else:
            logger.info(
                "Best: %s  (reward=%+.6f)",
                best_window_agent_path.name, best_val_reward,
            )

        # Warm-start next window from the best agent of this window
        best_agent_path = best_window_agent_path

    logger.info("Training complete.")


if __name__ == "__main__":
    train_pipeline()
