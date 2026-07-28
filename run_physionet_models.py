import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT_DIR / "Experiments" / "PhysioNet_Results"


MODEL_CONFIGS = {
    # Table 4: autoregressive ODE-RNN.
    "ode_rnn": {
        "flags": ["--ode-rnn"],
        "latents": 20,
        "rec_dims": 40,       # Unused by standalone ODE-RNN.
        "rec_layers": 3,
        "gen_layers": 1,      # Unused by standalone ODE-RNN.
        "units": 50,
        "gru_units": 50,
        "supports_extrapolation": False,
    },

    # Table 5: Latent ODE with ODE-RNN recognition network.
    "latent_ode_odernn": {
        "flags": [
            "--latent-ode",
            "--z0-encoder",
            "odernn",
        ],
        "latents": 20,
        "rec_dims": 40,
        "rec_layers": 3,
        "gen_layers": 3,
        "units": 50,
        "gru_units": 50,
        "supports_extrapolation": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Run PhysioNet autoregressive and encoder-decoder experiments"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=[
            "ode_rnn",
            "latent_ode_odernn",
            "latent_ode_rnn",
            "rnn_vae",
        ],
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["interpolation", "extrapolation"],
        default=["interpolation", "extrapolation"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1991, 1992, 1993, 1994, 1995],
    )

    parser.add_argument("-n", type=int, default=4000)
    parser.add_argument("--niters", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument(
        "--quantization",
        type=float,
        default=1.0 / 60.0,
        help="Timestamp quantization in hours; 1/60 is one minute.",
    )

    parser.add_argument(
        "--classif",
        action="store_true",
        help="Train the in-hospital mortality classifier.",
    )

    parser.add_argument(
        "--poisson",
        action="store_true",
        help="Add the Poisson observation-process likelihood.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    return parser.parse_args()


def build_command(args, model_name, task, seed, run_dir):
    config = MODEL_CONFIGS[model_name]

    command = [
        sys.executable,
        "-m",
        "Experiments.run_models",

        "--dataset",
        "physionet",

        "-n",
        str(args.n),

        "--niters",
        str(args.niters),

        "--batch-size",
        str(args.batch_size),

        "--lr",
        str(args.lr),

        "--quantization",
        str(args.quantization),

        "--random-seed",
        str(seed),

        "--save",
        str(run_dir),

        "--latents",
        str(config["latents"]),

        "--rec-dims",
        str(config["rec_dims"]),

        "--rec-layers",
        str(config["rec_layers"]),

        "--gen-layers",
        str(config["gen_layers"]),

        "--units",
        str(config["units"]),

        "--gru-units",
        str(config["gru_units"]),
    ]

    command.extend(config["flags"])

    if task == "extrapolation":
        command.append("--extrap")

    if args.classif:
        command.append("--classif")

    if args.poisson:
        if model_name == "latent_ode_odernn":
            command.append("--poisson")
        else:
            print(
                f"Poisson flag ignored for model {model_name}; "
                "it is only enabled for Latent ODE."
            )

    return command


def run_command(command, env, dry_run):
    print("Command:")
    print("  " + " ".join(str(item) for item in command))

    if dry_run:
        return 0

    result = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env,
    )

    return result.returncode


def load_metrics(run_dir):
    metrics_path = run_dir / "final_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Run completed without producing {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    seen = set()

    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(rows, field):
    values = []

    for row in rows:
        value = row.get(field)

        if value is None:
            continue

        value = float(value)

        if math.isfinite(value):
            values.append(value)

    return np.asarray(values, dtype=float)


def summarize_field(rows, field):
    values = numeric_values(rows, field)

    if values.size == 0:
        return {
            f"{field}_mean": None,
            f"{field}_std": None,
            f"{field}_stderr": None,
        }

    mean = float(values.mean())

    if values.size == 1:
        std = 0.0
        stderr = 0.0
    else:
        std = float(values.std(ddof=1))
        stderr = float(std / np.sqrt(values.size))

    return {
        f"{field}_mean": mean,
        f"{field}_std": std,
        f"{field}_stderr": stderr,
    }


def make_summary_rows(result_rows):
    summary_rows = []

    groups = sorted({
        (
            row["model"],
            row["task"],
            row["classification"],
            row["poisson"],
        )
        for row in result_rows
    })

    for model, task, classification, poisson in groups:
        group_rows = [
            row
            for row in result_rows
            if row["model"] == model
            and row["task"] == task
            and row["classification"] == classification
            and row["poisson"] == poisson
        ]

        summary = {
            "model": model,
            "task": task,
            "classification": classification,
            "poisson": poisson,
            "n_runs": len(group_rows),
        }

        for field in [
            "test_mse",
            "test_mse_x1e3",
            "test_loss",
            "test_likelihood",
            "test_auc",
            "test_ce_loss",
        ]:
            summary.update(summarize_field(group_rows, field))

        summary_rows.append(summary)

    return summary_rows


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT_DIR)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    result_rows = []

    planned_runs = []

    for model_name in args.models:
        config = MODEL_CONFIGS[model_name]

        for task in args.tasks:
            if (
                task == "extrapolation"
                and not config["supports_extrapolation"]
            ):
                print(
                    f"Skipping {model_name} extrapolation: "
                    "the autoregressive implementation does not support it."
                )
                continue

            for seed in args.seeds:
                planned_runs.append((model_name, task, seed))

    print(f"Repository: {ROOT_DIR}")
    print(f"Results:    {args.results_dir}")
    print(f"Runs:       {len(planned_runs)}")
    print()

    for run_index, (model_name, task, seed) in enumerate(
        planned_runs,
        start=1,
    ):
        run_label = (
            f"{model_name}"
            f"_{task}"
            f"_seed-{seed}"
        )

        if args.classif:
            run_label += "_mortality"

        if args.poisson and model_name == "latent_ode_odernn":
            run_label += "_poisson"

        run_dir = args.results_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        command = build_command(
            args,
            model_name,
            task,
            seed,
            run_dir,
        )

        print("=" * 80)
        print(f"Run {run_index}/{len(planned_runs)}: {run_label}")
        print("=" * 80)

        return_code = run_command(
            command,
            env,
            args.dry_run,
        )

        if args.dry_run:
            continue

        if return_code != 0:
            if args.continue_on_error:
                print(f"Run failed with exit code {return_code}")
                continue

            raise SystemExit(return_code)

        metrics = load_metrics(run_dir)
        metrics["run_label"] = run_label
        metrics["save_dir"] = str(run_dir)

        result_rows.append(metrics)

        # Update files after every completed run so interrupted sweeps
        # preserve all completed results.
        write_csv(
            args.results_dir / "aggregate_results.csv",
            result_rows,
        )

        write_csv(
            args.results_dir / "aggregate_summary.csv",
            make_summary_rows(result_rows),
        )

        print(
            f"Completed: test_mse={metrics.get('test_mse')}, "
            f"test_auc={metrics.get('test_auc')}"
        )
        print()

    if not args.dry_run:
        write_csv(
            args.results_dir / "aggregate_results.csv",
            result_rows,
        )

        write_csv(
            args.results_dir / "aggregate_summary.csv",
            make_summary_rows(result_rows),
        )

    print(f"Results saved under {args.results_dir}")


if __name__ == "__main__":
    main()