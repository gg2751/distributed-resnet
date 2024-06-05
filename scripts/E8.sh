#!/bin/bash
#SBATCH --job-name=E8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E8: Large Batch Training
python ./eval.py --batch_size=8192 --num_gpus=2 --get_bandwidth=True
python ./eval.py --batch_size=8192 --num_gpus=4 --get_bandwidth=True