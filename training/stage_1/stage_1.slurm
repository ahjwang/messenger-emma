#!/bin/bash

#SBATCH --job-name emma_stage-1
#SBATCH --output=slurm/%x-%j

#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --mem=4G
#SBATCH -t 12:00:00

"$@"