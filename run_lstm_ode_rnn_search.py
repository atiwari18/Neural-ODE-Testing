import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"

LSTM_SCRIPT = EXPERIMENTS_DIR / "lstm_spiral_experiment.py"
ODE_RNN_SCRIPT = EXPERIMENTS_DIR / "run_models.py"

RESULTS_DIR = ROOT_DIR / "Experiments" / "Uniform_Search_Results"
SHARED_SPIRAL_PATH = ROOT_DIR / "Experiments" / "shared_spiral_dataset_scoring_pi.pt"

LSTM_GRID = [
    # epochs, lr, hidden_dim, num_layers, teacher_forcing
    #(1, 1e-3, 1, 1, 0.5)

    # Small / fast baselines
    (200, 1e-3, 32, 1, 0.5),
    (200, 5e-4, 32, 1, 0.5),

    # Medium capacity, shallow
    (300, 1e-3, 64, 1, 0.5),
    (300, 5e-4, 64, 1, 0.5),

    # Medium capacity, deeper
    (300, 1e-3, 64, 2, 0.5),
    (300, 5e-4, 64, 2, 0.5),

    # Larger hidden state
    (500, 1e-3, 96, 1, 0.5),
    (500, 5e-4, 96, 1, 0.5),

    # Larger + deeper
    (500, 1e-3, 96, 2, 0.5),
    (500, 5e-4, 96, 2, 0.5),

    # Explicit depth stress tests
    (500, 1e-3, 64, 3, 0.5),
    (500, 5e-4, 96, 3, 0.5),
]

ODE_RNN_GRID = [
    # niters, lr, latents, rec_dims, units, gru_units, rec_layers, gen_layers
    #(1, 1e-2, 1, 1, 1,  1,  1, 1)
    # Small / fast baselines
    (1000, 1e-2, 4, 20, 64,  64,  1, 1),
    (1000, 5e-3, 4, 20, 64,  64,  1, 1),

    # Medium latent capacity
    (1500, 5e-3, 6, 20, 64,  64,  1, 1),
    (1500, 1e-3, 6, 20, 64,  64,  1, 1),

    # Recognition/generative depth tests
    (1500, 5e-3, 6, 20, 64,  64,  2, 1),
    (1500, 5e-3, 6, 20, 64,  64,  1, 2),

    # Larger latent/capacity
    (3000, 5e-3, 8, 30, 64,  64,  1, 1),
    (3000, 1e-3, 8, 30, 64,  64,  1, 1),

    # Wider ODE/GRU networks
    (3000, 5e-3, 6, 30, 100, 100, 1, 1),
    (3000, 1e-3, 6, 30, 100, 100, 1, 1),

    # Deeper ODE function tests
    (3000, 5e-3, 8, 30, 100, 100, 2, 2),
    (3000, 1e-3, 8, 30, 100, 100, 3, 2),
]

def parse_args():
    parser = argparse.ArgumentParser("LSTM vs. ODE-RNN hyperparameter search")
    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=["all", "lstm", "ode-rnn"], default="all")
    return parser.parse_args()

def run_command(cmd, cwd, env, dry_run):
    print(" ".join(str(x) for x in cmd))
    if dry_run: 
        return
    
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    
def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "manifest.csv"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    rows = []

    if args.only in ("all", "lstm"):
        for idx, (epochs, lr, hidden_dim, num_layers, teacher_forcing) in enumerate(LSTM_GRID, start=1):
            label = (
                f"lstm_run-{idx:02d}"
                f"_epochs-{epochs}"
                f"_lr-{lr}"
                f"_hidden-{hidden_dim}"
                f"_layers-{num_layers}"
            )
            run_dir = RESULTS_DIR / label
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(LSTM_SCRIPT),
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--hidden-dim", str(hidden_dim),
                "--num-layers", str(num_layers),
                "--teacher-forcing", str(teacher_forcing),
                "--seed", str(args.seed),
                "--save-dir", str(run_dir),

                # Match the shared ODE-RNN spiral dataset config.
                "--nspiral", "1000",
                "--ntotal", "1000",
                "--obs-len", "15",
                "--pred-len", "800",
                "--stop", "18.85",
                "--noise-std", "0.1",
                "--irregular_spiral",
                "--irregular_window_time", "3.141592653589793",
                "--n_trials", "100",
                "--shared_spiral_path", str(SHARED_SPIRAL_PATH),

                "--plot-every", str(epochs),
            ]

            rows.append({
                "model": "lstm",
                "run": idx,
                "label": label,
                "epochs_or_niters": epochs,
                "lr": lr,
                "hidden_dim_or_latents": hidden_dim,
                "num_layers": num_layers,
                "teacher_forcing": teacher_forcing,
                "rec_dims": "",
                "units": "",
                "gru_units": "",
                "save_dir": run_dir,
            })

            print(f"\n=== LSTM {idx}/{len(LSTM_GRID)}: {label} ===")
            run_command(cmd, ROOT_DIR, env, args.dry_run)

    if args.only in ("all", "ode-rnn"):
        for idx, (niters, lr, latents, rec_dims, units, gru_units, rec_layers, gen_layers) in enumerate(ODE_RNN_GRID, start=1):
            label = (
                f"ode_rnn_run-{idx:02d}"
                f"_niters-{niters}"
                f"_lr-{lr}"
                f"_latents-{latents}"
                f"_rec-{rec_dims}"
                f"_units-{units}"
                f"_gru-{gru_units}"
            )
            run_dir = RESULTS_DIR / label
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(ODE_RNN_SCRIPT),
                "--dataset", "spiral",
                "--latent-ode",
                "--z0-encoder", "odernn",
                "--spiral",
                "-n", "1000",
                "-b", "64",
                "--niters", str(niters),
                "--lr", str(lr),
                "--latents", str(latents),
                "--rec-dims", str(rec_dims),
                "--rec-layers", str(rec_layers),
                "--gen-layers", str(gen_layers),
                "--units", str(units),
                "--gru-units", str(gru_units),
                "--timepoints", "15",
                "--max-t", "18.85",
                "--noise-weight", "0.1",
                "--ntotal", "1000",
                "--irregular_spiral",
                "--irregular_window_time", "3.141592653589793",
                "--shared_spiral_path", str(SHARED_SPIRAL_PATH),
                "--random-seed", str(args.seed),
                "--save", str(run_dir),
            ]

            rows.append({
                "model": "ode-rnn",
                "run": idx,
                "label": label,
                "epochs_or_niters": niters,
                "lr": lr,
                "hidden_dim_or_latents": latents,
                "num_layers": "",
                "teacher_forcing": "",
                "rec_dims": rec_dims,
                "units": units,
                "gru_units": gru_units,
                "save_dir": run_dir,
            })

            print(f"\n=== ODE-RNN {idx}/{len(ODE_RNN_GRID)}: {label} ===")
            run_command(cmd, ROOT_DIR, env, args.dry_run)

    if rows:
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()