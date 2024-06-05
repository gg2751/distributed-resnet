#!/bin/bash
#SBATCH --job-name=E5
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E5: Computational Efficiency w.r.t Batch Size
python ./eval.py --batch_size=128
python ./eval.py --batch_size=512
python ./eval.py --batch_size=2048
python ./eval.py --batch_size=8192
