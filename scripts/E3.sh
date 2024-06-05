#!/bin/bash
#SBATCH --job-name=E3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E3: GPU vs CPU
python ./eval.py --batch_size=32 --num_workers=4 --use-cuda=False
python ./eval.py --batch_size=32 --num_workers=4
