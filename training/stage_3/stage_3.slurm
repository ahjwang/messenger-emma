#!/bin/bash

#SBATCH --job-name emma_stage-3
#SBATCH --output=slurm/%x-%j

#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --mem=8G
#SBATCH -t 72:00:00
#SBATCH -x node718,node030,node004,node015

"$@"