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

module load python
module load cuda

cd "$SLURM_SUBMIT_DIR/Experiments" || exit 1

source .venv/bin/activate
mkdir -p logs

python3 run_experiments.py