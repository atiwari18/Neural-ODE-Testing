#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --job-name="latent_ode_experiment"
#SBATCH --partition=short
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Experiments/logs/slurm_%j.out
#SBATCH --error=Experiments/logs/slurm_%j.err

mkdir -p Experiments/logs

module load python
module load cuda

# Stay in the project root (SLURM_SUBMIT_DIR) — run_experiments.py lives here
cd "$SLURM_SUBMIT_DIR" || exit 1

# Activate venv from the project root
source .venv/bin/activate

python3 run_experiments.py