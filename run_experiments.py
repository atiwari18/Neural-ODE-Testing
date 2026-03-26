"""
Usage (from Neural-ODE-Testing/):
    python run_experiments.py
    python run_experiments.py --visualize True
"""
import subprocess
import sys
import argparse
import os
from pathlib import Path

# Root of the repo — where this script lives
ROOT_DIR = Path(__file__).resolve().parent

# Experiments/ subfolder — where anneal_experiment_alternate.py and models/ live.
# cwd must be set here so that `from models.latent_ode import ...` resolves.
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"

# Sanity check: fail fast with a clear message if the layout looks wrong
if not (EXPERIMENTS_DIR / "anneal_experiment_alternate.py").exists():
    print(f"ERROR: Could not find anneal_experiment_alternate.py at {EXPERIMENTS_DIR}")
    print( "       Make sure run_experiments.py is in the project root and")
    print( "       Experiments/anneal_experiment.py exists.")
    sys.exit(1)

if not (ROOT_DIR / "models").exists():
    print(f"ERROR: Could not find models/ folder at {ROOT_DIR / 'models'}")
    print( "       The models/ folder must be inside project root/.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser("Run all experiments")
    parser.add_argument('--visualize', type=str, default='True')
    parser.add_argument('--seed',      type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    NITERS_LIST    = [10000, 20000]
    KL_ANNEAL_LIST = [True, False]
    LR_LIST        = [0.01, 0.005] 

    total = len(NITERS_LIST) * len(KL_ANNEAL_LIST) * len(LR_LIST)
    count = 0

    print(f"Experiments dir : {EXPERIMENTS_DIR}")
    print(f"Total runs      : {total}\n")

    for niters in NITERS_LIST:
        for kl_anneal in KL_ANNEAL_LIST:
            for lr in LR_LIST:
                count += 1
                label = "kl_anneal" if kl_anneal else "no_kl_anneal"

                print("=" * 60)
                print(f"  Run {count}/{total}: niters={niters}  lr={lr}  {label}")
                print("=" * 60)

                cmd = [
                    sys.executable,
                    #PATH TO SCRIPT CHANGE FOR RELEVANT EXPERIMENT
                    str(EXPERIMENTS_DIR / "anneal_experiment_alternate.py"),
                    "--niters",    str(niters),
                    "--lr",        str(lr),
                    "--seed",      str(args.seed),
                    "--kl_anneal", "True" if kl_anneal else "False",
                    "--visualize", args.visualize,
                ]

                # cwd=EXPERIMENTS_DIR so relative file I/O (logs, figs) land there.
                # PYTHONPATH explicitly adds it to the import search path —
                # cwd alone is not guaranteed to be on sys.path on Windows.
                env = os.environ.copy()

                # Add project root so "models" can be imported
                env["PYTHONPATH"] = (
                    str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
                )

                result = subprocess.run(
                    cmd,
                    cwd=EXPERIMENTS_DIR,
                    env=env
                )

                if result.returncode != 0:
                    print(f"\nRun {count}/{total} failed (exit code {result.returncode}). Stopping.")
                    sys.exit(result.returncode)

                print()

    print(f"All {total} experiments complete.")