#!/bin/bash
# Job name:
#SBATCH --job-name=benchmarks
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=vm-small
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
#SBATCH --mem=0   
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=16
#
# Wall clock limit: HH:MM:SS
#SBATCH --time=47:30:00
#
python lyap.py
