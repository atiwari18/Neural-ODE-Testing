#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --job-name="latent_ode_experiment"
#SBATCH --partition=short
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100"
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err