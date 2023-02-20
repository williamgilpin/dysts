#!/bin/bash
# Job name:
#SBATCH --job-name=benchmarks
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=normal
#
# Job progress file
#SBATCH --output=benchmarks_timing.out
#
# Error file:
#SBATCH --error=benchmarks_timing.err      
#
# Request one node:
#SBATCH --nodes=1
#
# memory per node:
#SBATCH --mem=32000   
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit: HH:MM:SS
#SBATCH --time=2:30:00
#
## Command(s) to run (example):
conda init bash
conda info --envs
echo $PATH
source ~/work/miniconda3/etc/profile.d/conda.sh
conda activate dysts
echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX
python timing_benchmarks2.py
