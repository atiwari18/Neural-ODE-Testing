import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from dataset.lstm_dataset import make_synthetic_kt_timesteps


ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT_DIR / "Experiments"
RUN_SCRIPT = EXPERIMENTS_DIR / "dkt_ode_rnn_synthetic.py"

DATA_DIR = ROOT_DIR / "syntheticKT"
RESULTS_DIR = ROOT_DIR / "SyntheticDKTResults" / "ODE_RNN_Results_Irregular_48"


def parse_args():
    parser = argparse.ArgumentParser("Run ODE-RNN DKT on synthetic KT CSVs")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--concepts", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--versions", nargs="+", type=int, default=list(range(20)))

    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=400)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--ode-units", type=int, default=64)
    parser.add_argument("--gru-units", type=int, default=64)
    parser.add_argument("--ode-method", type=str, default="euler")

    #Irregular
    parser.add_argument("--irregular-timestamps", action="store_true")
    parser.add_argument("--irregular-window-time", type=float, default=None)
    parser.add_argument("--irregular-n-trials", type=int, default=250)
    parser.add_argument("--time-steps-path", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=50)

    return parser.parse_args()

def get_or_create_time_steps_file(args):
    if not args.irregular_timestamps:
        return None

    if args.time_steps_path is not None:
        path = Path(args.time_steps_path)
    else:
        window = args.irregular_window_time

        if window is None:
            window = float(args.num_questions - 2)

        window_label = str(window).replace(".", "p")

        path = (
            EXPERIMENTS_DIR
            / (
                f"shared_synthetic_kt_irregular_times"
                f"_q{args.num_questions}"
                f"_window-{window_label}"
                f"_seed-{args.seed}.pt"
            )
        )

    if path.exists():
        print(f"Using existing shared time steps: {path}")
        return path

    np.random.seed(args.seed)

    time_steps = make_synthetic_kt_timesteps(
        num_input_steps=args.num_questions - 1,
        irregular=True,
        irregular_window_time=args.irregular_window_time,
        n_trials=args.irregular_n_trials,
    )

    payload = {
        "time_steps": time_steps,
        "num_questions": args.num_questions,
        "num_input_steps": args.num_questions - 1,
        "irregular": True,
        "irregular_window_time": args.irregular_window_time,
        "irregular_n_trials": args.irregular_n_trials,
        "seed": args.seed,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

    print(f"Saved shared irregular time steps: {path}")

    return path

def build_command(args, csv_path, run_dir, time_steps_path=None):
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--csv-path", str(csv_path),
        "--save-dir", str(run_dir),

        "--train-size", str(args.train_size),
        "--val-size", str(args.val_size),

        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),

        "--latent-dim", str(args.latent_dim),
        "--ode-units", str(args.ode_units),
        "--gru-units", str(args.gru_units),
        "--ode-method", str(args.ode_method),

        "--irregular-n-trials", str(args.irregular_n_trials),

        "--seed", str(args.seed),
    ]

    if args.irregular_timestamps:
        cmd.append("--irregular-timestamps")

    if args.irregular_window_time is not None:
        cmd.extend([
            "--irregular-window-time", 
            str(args.irregular_window_time)
        ])

    if time_steps_path is not None:
        cmd.extend(["--time-steps-path", str(time_steps_path)])

    return cmd

def run_command(cmd, cwd, env, dry_run):
    print("command:")
    print("  " + " ".join(str(x) for x in cmd))

    if dry_run:
        return

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_metrics(run_dir):
    metrics_path = run_dir / "final_metrics.json"

    with metrics_path.open("r") as f:
        return json.load(f)


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std_stderr(values):
    values = np.array(values, dtype=float)
    mean = values.mean()

    if len(values) < 2:
        return mean, 0.0, 0.0

    std = values.std(ddof=1)
    stderr = std / np.sqrt(len(values))

    return mean, std, stderr


if __name__ == "__main__":
    args = parse_args()

    if not RUN_SCRIPT.exists():
        print(f"ERROR: Could not find run script at {RUN_SCRIPT}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    time_steps_path = get_or_create_time_steps_file(args)

    manifest_rows = []
    result_rows = []

    total_runs = len(args.concepts) * len(args.versions)

    print(f"Run script  : {RUN_SCRIPT}")
    print(f"Data dir    : {DATA_DIR}")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Total runs  : {total_runs}")
    print()

    run_idx = 0

    for concepts in args.concepts:
        for version in args.versions:
            run_idx += 1

            csv_path = DATA_DIR / f"naive_c{concepts}_q50_s4000_v{version}.csv"

            if not csv_path.exists():
                print(f"Missing dataset, skipping: {csv_path}")
                continue

            time_label = "irregular" if args.irregular_timestamps else "regular"

            label = (
                f"c{concepts}"
                f"_v{version:02d}"
                f"_{time_label}"
                f"_epochs-{args.epochs}"
                f"_lr-{args.lr}"
                f"_latent-{args.latent_dim}"
                f"_ode-{args.ode_units}"
                f"_gru-{args.gru_units}"
            )

            run_dir = RESULTS_DIR / label
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_command(args, csv_path, run_dir, time_steps_path)

            manifest_rows.append({
                "model": "ode_rnn_dkt",
                "concepts": concepts,
                "version": version,
                "label": label,
                "csv_path": csv_path,
                "save_dir": run_dir,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "latent_dim": args.latent_dim,
                "ode_units": args.ode_units,
                "gru_units": args.gru_units,
                "ode_method": args.ode_method,
                "batch_size": args.batch_size,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "seed": args.seed,
                "irregular_timestamps": args.irregular_timestamps,
                "irregular_window_time": args.irregular_window_time,
                "irregular_n_trials": args.irregular_n_trials,
                "time_steps_path": time_steps_path if time_steps_path is not None else ""
            })

            print("=" * 80)
            print(f"Run {run_idx}/{total_runs}")
            print(f"label      : {label}")
            print(f"csv        : {csv_path}")
            print(f"output dir : {run_dir}")
            print("=" * 80)

            run_command(cmd, ROOT_DIR, env, args.dry_run)

            if not args.dry_run:
                metrics = load_metrics(run_dir)

                result_rows.append({
                    "model": "ode_rnn_dkt",
                    "concepts": concepts,
                    "version": version,
                    "label": label,

                    "best_epoch": metrics["best_epoch"],
                    "epochs_completed": metrics["epochs_completed"],

                    "best_train_loss": metrics["best_train_loss"],
                    "best_train_accuracy": metrics["best_train_accuracy"],
                    "best_train_auc": metrics["best_train_auc"],

                    "best_val_loss": metrics["best_val_loss"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "best_val_auc": metrics["best_val_auc"],

                    "test_loss": metrics["test_loss"],
                    "test_accuracy": metrics["test_accuracy"],
                    "test_auc": metrics["test_auc"],

                    "save_dir": run_dir,
                })

            print()

    manifest_path = RESULTS_DIR / "manifest.csv"
    write_csv(manifest_path, manifest_rows)

    if not args.dry_run:
        aggregate_results_path = RESULTS_DIR / "aggregate_results.csv"
        write_csv(aggregate_results_path, result_rows)

        summary_rows = []

        for concepts in args.concepts:
            rows = [row for row in result_rows if row["concepts"] == concepts]

            if not rows:
                continue

            test_loss_mean, test_loss_std, test_loss_stderr = mean_std_stderr(
                [row["test_loss"] for row in rows]
            )
            test_accuracy_mean, test_accuracy_std, test_accuracy_stderr = mean_std_stderr(
                [row["test_accuracy"] for row in rows]
            )
            test_auc_mean, test_auc_std, test_auc_stderr = mean_std_stderr(
                [row["test_auc"] for row in rows]
            )
            val_loss_mean, val_loss_std, val_loss_stderr = mean_std_stderr(
                [row["best_val_loss"] for row in rows]
            )
            val_auc_mean, val_auc_std, val_auc_stderr = mean_std_stderr(
                [row["best_val_auc"] for row in rows]
            )

            summary_rows.append({
                "model": "ode_rnn_dkt",
                "concepts": concepts,
                "n_runs": len(rows),

                "best_val_loss_mean": val_loss_mean,
                "best_val_loss_std": val_loss_std,
                "best_val_loss_stderr": val_loss_stderr,

                "best_val_auc_mean": val_auc_mean,
                "best_val_auc_std": val_auc_std,
                "best_val_auc_stderr": val_auc_stderr,

                "test_loss_mean": test_loss_mean,
                "test_loss_std": test_loss_std,
                "test_loss_stderr": test_loss_stderr,

                "test_accuracy_mean": test_accuracy_mean,
                "test_accuracy_std": test_accuracy_std,
                "test_accuracy_stderr": test_accuracy_stderr,

                "test_auc_mean": test_auc_mean,
                "test_auc_std": test_auc_std,
                "test_auc_stderr": test_auc_stderr,
            })

        aggregate_summary_path = RESULTS_DIR / "aggregate_summary.csv"
        write_csv(aggregate_summary_path, summary_rows)

        print(f"Manifest saved          : {manifest_path}")
        print(f"Aggregate results saved : {aggregate_results_path}")
        print(f"Aggregate summary saved : {aggregate_summary_path}")
    else:
        print(f"Dry run complete. Manifest preview saved: {manifest_path}")

    print(f"Results saved under: {RESULTS_DIR}")