#!/bin/bash
# Job name:
#SBATCH --job-name=benchmarks
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=gpu-a100
#
# Job progress file
#SBATCH --output=benchmarks.out
#
# Error file:
#SBATCH --error=benchmarks.err      
#
# Request one node:
#SBATCH --nodes=1
#
# memory per node:
#SBATCH --mem=64000   
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=2
#
# Wall clock limit: HH:MM:SS
#SBATCH --time=47:29:00
#
## Command(s) to run (example):
#conda init bash
#conda info --envs
#echo $PATH
#source ~/work/miniconda3/etc/profile.d/conda.sh
#conda activate dysts
#echo $CONDA_DEFAULT_ENV
#echo $CONDA_PREFIX
python find_hyperparameters_multivariate.py
