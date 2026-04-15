import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"
RESULTS_DIR = ROOT_DIR / "ODE-RNN_Results"

RUN_SCRIPT = EXPERIMENTS_DIR / "run_models.py"


def parse_args():
    parser = argparse.ArgumentParser("Run Latent ODE experiment grid")
    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="spiral")
    parser.add_argument("--ntotal", type=int, default=None)
    return parser.parse_args()


def build_command(args, run_dir, niters, lr, latents, timepoints, noise_weight):
    shared_spiral_path = ROOT_DIR / "Experiments" / "shared_spiral_dataset_scoring_pi.pt"

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),

        "--dataset", "spiral",
        "--latent-ode",
        "--spiral",

        "-n", "1000",
        "-b", "64",
        "--niters", str(niters),
        "--lr", str(lr),
        "--timepoints", str(timepoints),
        "--max-t", "18.85",
        "--noise-weight", str(noise_weight),
        "--latents", str(latents),
        "--irregular_spiral",
        "--irregular_window_time", str(np.pi),


        # Use one shared saved dataset so different models see the same spirals.
        "--shared_spiral_path", str(shared_spiral_path),

        "--save", str(run_dir),
    ]

    if args.ntotal is not None:
        cmd.extend(["--ntotal", str(args.ntotal)])

    return cmd


if __name__ == "__main__":
    args = parse_args()

    if not RUN_SCRIPT.exists():
        print(f"ERROR: Could not find run_models.py at {RUN_SCRIPT}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Hyperparameter grid
    NITERS_LIST = [3000]
    LR_LIST = [1e-2]
    LATENT_DIM_LIST = [6]
    TIMEPOINTS_LIST = [15]
    NOISE_WEIGHT_LIST = [0.1]

    grid = list(itertools.product(
        NITERS_LIST,
        LR_LIST,
        LATENT_DIM_LIST,
        TIMEPOINTS_LIST,
        NOISE_WEIGHT_LIST,
    ))

    print(f"Run script   : {RUN_SCRIPT}")
    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Total runs   : {len(grid)}")
    print()

    for run_idx, (niters, lr, latents, timepoints, noise_weight) in enumerate(grid, start=1):
        label = (
            f"niters-{niters}"
            f"_lr-{lr}"
            f"_latents-{latents}"
            f"_tp-{timepoints}"
            f"_noise-{noise_weight}-irregular-scoring"
        )

        run_dir = RESULTS_DIR / label
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            args=args,
            run_dir=run_dir,
            niters=niters,
            lr=lr,
            latents=latents,
            timepoints=timepoints,
            noise_weight=noise_weight,
        )

        print("=" * 80)
        print(f"Run {run_idx}/{len(grid)}")
        print(f"label      : {label}")
        print(f"output dir : {run_dir}")
        print("command    :")
        print("  " + " ".join(cmd))
        print("=" * 80)

        # if args.dry_run:
        #     print()
        #     continue

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