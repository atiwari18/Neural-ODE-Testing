import subprocess
import sys
import argparse
import os
from pathlib import Path

# Root of the repo
ROOT_DIR = Path(__file__).resolve().parent

# Experiments/ subfolder (where ode-rnn_experiment.py lives)
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"

# All outputs land here; one sub-directory per run
RESULTS_DIR = ROOT_DIR / "ODE-RNN_Results"

# Sanity check
if not (EXPERIMENTS_DIR / "ode-rnn_experiment.py").exists():
    print(f"ERROR: Could not find ode-rnn_experiment.py at {EXPERIMENTS_DIR}")
    print("       Make sure you placed ode-rnn_experiment.py inside Experiments/")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser("Run Latent ODE spiral experiments")
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    #HYPERPARAMETER GRID
    NITERS_LIST     = [1000, 3000, 5000]
    LR_LIST         = [0.005]
    LATENT_DIM_LIST = [4, 6]
    KL_COEF_LIST    = [0.5, 1.0]
    total = (len(NITERS_LIST) * len(LR_LIST)
             * len(LATENT_DIM_LIST) * len(KL_COEF_LIST))
    count = 0

    print(f"Experiments dir : {EXPERIMENTS_DIR}")
    print(f"Results dir     : {RESULTS_DIR}")
    print(f"Total runs      : {total}\n")

    for niters in NITERS_LIST:
        for lr in LR_LIST:
            for latent_dim in LATENT_DIM_LIST:
                for kl_coef in KL_COEF_LIST:
                    count += 1
                    label = (f"niters-{niters}_lr-{lr}_latent-{latent_dim}"
                             f"_kl-{kl_coef}")
                    run_dir = RESULTS_DIR / label
                    run_dir.mkdir(parents=True, exist_ok=True)

                    print("=" * 70)
                    print(f"  Run {count}/{total}: niters={niters}  lr={lr}"
                          f"  latent_dim={latent_dim}  kl_coef={kl_coef}")
                    print(f"  Output dir: {run_dir}")
                    print("=" * 70)

                    cmd = [
                        sys.executable,
                        str(EXPERIMENTS_DIR / "ode-rnn_experiment.py"),
                        "--niters",     str(niters),
                        "--lr",         str(lr),
                        "--latent_dim", str(latent_dim),
                        "--kl_coef",    str(kl_coef),
                        "--seed",       str(args.seed),
                    ]

                    if args.visualize:
                        cmd.append("--visualize")

                    env = os.environ.copy()
                    env["PYTHONPATH"] = (str(ROOT_DIR)
                                        + os.pathsep
                                        + env.get("PYTHONPATH", ""))

                    result = subprocess.run(
                        cmd,
                        cwd=str(run_dir),   # <-- was EXPERIMENTS_DIR
                        env=env,
                    )

                    if result.returncode != 0:
                        print(f"\nRun {count}/{total} failed "
                              f"(exit code {result.returncode}). Stopping.")
                        sys.exit(result.returncode)

                    print()

    print(f"All {total} experiments complete.")
    print(f"Results saved under: {RESULTS_DIR}")