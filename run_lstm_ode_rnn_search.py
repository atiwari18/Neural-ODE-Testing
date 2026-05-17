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