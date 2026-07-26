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
RUN_SCRIPT = EXPERIMENTS_DIR / "dkt_lstm_synthetic.py"

DATA_DIR = ROOT_DIR / "syntheticKT"
RESULTS_DIR = ROOT_DIR / "SyntheticDKTResults" / "LSTM_Results_Irregular_48"


def parse_args():
    parser = argparse.ArgumentParser("Run LSTM DKT on synthetic KT CSVs")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--concepts", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--versions", nargs="+", type=int, default=list(range(20)))

    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--include-delta-t", action="store_true")
    parser.add_argument("--irregular-timestamps", action="store_true")
    parser.add_argument("--irregular-window-time", type=float, default=None)
    parser.add_argument("--irregular-n-trials", type=int, default=100)
    parser.add_argument("--time-steps-path", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=50)

    return parser.parse_args()

def get_or_create_time_steps_file(args):
    if not args.include_delta_t:
        return None

    if args.time_steps_path is not None:
        return Path(args.time_steps_path)

    if not args.irregular_timestamps:
        return None

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
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--irregular-n-trials", str(args.irregular_n_trials),
    ]

    if args.include_delta_t:
        cmd.append("--include-delta-t")

    if args.irregular_timestamps:
        cmd.append("--irregular-timestamps")

    if args.irregular_window_time is not None:
        cmd.extend(["--irregular-window-time", str(args.irregular_window_time)])

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

            timing_label = "regular"

            if args.include_delta_t: 
                timing_label = "delta_t_irregular" if args.irregular_timestamps else "regular"

            label = (
                f"c{concepts}"
                f"_v{version:02d}"
                f"_{timing_label}"
                f"_epochs-{args.epochs}"
                f"_lr-{args.lr}"
                f"_hidden-{args.hidden_dim}"
                f"_layers-{args.num_layers}"
            )

            run_dir = RESULTS_DIR / label
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_command(args, csv_path, run_dir, time_steps_path)

            manifest_rows.append({
                "model": "lstm_dkt",
                "concepts": concepts,
                "version": version,
                "label": label,
                "csv_path": csv_path,
                "save_dir": run_dir,
                "epochs": args.epochs,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "train_size": args.train_size,
                "seed": args.seed,
                "include_delta_t": args.include_delta_t,
                "irregular_timestamps": args.irregular_timestamps,
                "irregular_window_time": args.irregular_window_time,
                "irregular_n_trials": args.irregular_n_trials,
                "time_steps_path": time_steps_path if time_steps_path is not None else "",
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
                    "model": "lstm_dkt",
                    "concepts": concepts,
                    "version": version,
                    "label": label,

                    "best_epoch": metrics["best_epoch"],

                    "best_train_loss": metrics["best_train_loss"],
                    "best_train_accuracy": metrics["best_train_accuracy"],
                    "best_train_auc": metrics["best_train_auc"],

                    "best_val_loss": metrics["best_val_loss"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "best_val_auc": metrics["best_val_auc"],

                    "test_loss": metrics["test_loss"],
                    "test_accuracy": metrics["test_accuracy"],
                    "test_auc": metrics["test_auc"],

                    "include_delta_t": metrics.get("include_delta_t", ""),
                    "irregular_timestamps": metrics.get("irregular_timestamps", ""),
                    "irregular_window_time": metrics.get("irregular_window_time", ""),
                    "time_steps_path": metrics.get("time_steps_path", ""),
                    "input_dim": metrics.get("input_dim", ""),

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

            test_losses = np.array([row["test_loss"] for row in rows], dtype=float)
            test_accs = np.array([row["test_accuracy"] for row in rows], dtype=float)
            test_aucs = np.array([row["test_auc"] for row in rows], dtype=float)

            summary_rows.append({
                "model": "lstm_dkt",
                "concepts": concepts,
                "n_runs": len(rows),
                "test_loss_mean": test_losses.mean(),
                "test_loss_std": test_losses.std(ddof=1),
                "test_loss_stderr": test_losses.std(ddof=1) / np.sqrt(len(rows)),
                "test_accuracy_mean": test_accs.mean(),
                "test_accuracy_std": test_accs.std(ddof=1),
                "test_accuracy_stderr": test_accs.std(ddof=1) / np.sqrt(len(rows)),
                "test_auc_mean": test_aucs.mean(),
                "test_auc_std": test_aucs.std(ddof=1),
                "test_auc_stderr": test_aucs.std(ddof=1) / np.sqrt(len(rows)),
            })

        aggregate_summary_path = RESULTS_DIR / "aggregate_summary.csv"
        write_csv(aggregate_summary_path, summary_rows)

        print(f"Manifest saved          : {manifest_path}")
        print(f"Aggregate results saved : {aggregate_results_path}")
        print(f"Aggregate summary saved : {aggregate_summary_path}")
    else:
        print(f"Dry run complete. Manifest preview saved: {manifest_path}")

    print(f"Results saved under: {RESULTS_DIR}")