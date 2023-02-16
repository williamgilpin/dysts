#!/bin/bash
# Job name:
#SBATCH --job-name=data-limited-small
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=vm-small
#
# Job progress file
#SBATCH --output=benchmarks_small.out
#
# Error file:
#SBATCH --error=benchmarks_small.err      
#
# Request one node:
#SBATCH --nodes=1
#
# memory per node:
#SBATCH --mem=0  
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=2
#
# Wall clock limit: HH:MM:SS
#SBATCH --time=47:30:00
#
## Command(s) to run (example):
python multivariate_data_limited2.py
