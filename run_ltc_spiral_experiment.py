import argparse 
import itertools
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"
RESULTS_DIR = ROOT_DIR / "LTC_Results"
RUN_SCRIPT = EXPERIMENTS_DIR / "run_ltc_spiral.py"

def parse_args():
    parser = argparse.ArgumentParser("Run LTC experiment grid")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ntotal", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=800)
    return parser.parse_args()

def build_command(args, run_dir, epochs, lr, hidden_dim, noise_weight, timepoints):
    shared_spiral_path = ROOT_DIR / "Experiments" / "shared_spiral_dataset_scoring_pi.pt"

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--hidden-dim", str(hidden_dim),
        "--noise-std", str(noise_weight),
        "--obs-len", str(timepoints),
        "--seed", str(args.seed),
        "--shared_spiral_path", str(shared_spiral_path),
        "--save-dir", str(run_dir),
        "--irregular_spiral",
        "--irregular_window_time", str(np.pi),
    ]

    if args.ntotal is not None:
        cmd.extend(["--ntotal", str(args.ntotal)])
        cmd.extend(["--pred_len", str(args.pred_len)])

    return cmd

if __name__ == "__main__":
    args = parse_args()

    if not RUN_SCRIPT.exists():
        print(f"ERROR: Could not find run_ltc_experiment.py at {RUN_SCRIPT}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    EPOCHS_LIST = [200]
    LR_LIST = [1e-3]
    HIDDEN_DIM_LIST = [32]
    TIMEPOINTS_LIST = [15]
    NOISE_WEIGHT_LIST = [0.1]

    grid = list(itertools.product(
        EPOCHS_LIST,
        LR_LIST,
        HIDDEN_DIM_LIST,
        TIMEPOINTS_LIST,
        NOISE_WEIGHT_LIST,
    ))

    print(f"Run script   : {RUN_SCRIPT}")
    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Total runs   : {len(grid)}")
    print()

    for run_idx, (epochs, lr, hidden_dim, timepoints, noise_weight) in enumerate(grid, start=1):
        label = (
            f"ltc-epochs-{epochs}"
            f"_lr-{lr}"
            f"_hidden-{hidden_dim}"
            f"_tp-{timepoints}"
            f"_noise-{noise_weight}-irregular"
        )

        run_dir = RESULTS_DIR / label
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(args, run_dir, epochs, lr, hidden_dim, noise_weight, timepoints)

        print("=" * 80)
        print(f"Run {run_idx}/{len(grid)}")
        print(f"label      : {label}")
        print(f"output dir : {run_dir}")
        print("command    :")
        print("  " + " ".join(cmd))
        print("=" * 80)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            env=env,
        )

        if result.returncode != 0:
            print(f"\nRun failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

        print()

    print(f"All {len(grid)} runs completed.")
    print(f"Results saved under: {RESULTS_DIR}")