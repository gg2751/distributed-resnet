#!/bin/bash
#SBATCH --job-name=E7
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E7: Computation vs Communication
python ./eval.py --batch_size=64 --num_gpus=2 --get_bandwidth=True
python ./eval.py --batch_size=64 --num_gpus=4 --get_bandwidth=True
python ./eval.py --batch_size=128 --num_gpus=2 --get_bandwidth=True
python ./eval.py --batch_size=128 --num_gpus=4 --get_bandwidth=True
python ./eval.py --batch_size=512 --num_gpus=2 --get_bandwidth=True
python ./eval.py --batch_size=512 --num_gpus=4 --get_bandwidth=True
python ./eval.py --batch_size=2048 --num_gpus=2 --get_bandwidth=True
python ./eval.py --batch_size=2048 --num_gpus=4 --get_bandwidth=True

