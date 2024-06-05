#!/bin/bash
#SBATCH --job-name=E2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E2: Profiling
python ./eval.py --batch_size=32 --num_workers=1
python ./eval.py --batch_size=32 --num_workers=1
