#!/bin/bash
#SBATCH --job-name=E6
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E6: Speedup Measurement
python ./eval.py --num_gpus=2
python ./eval.py --batch_size=128 --num_gpus=2
python ./eval.py --batch_size=512 --num_gpus=2
python ./eval.py --batch_size=2048 --num_gpus=2 
python ./eval.py --batch_size=8192 --num_gpus=2 

python ./eval.py --num_gpus=4
python ./eval.py --batch_size=128 --num_gpus=4
python ./eval.py --batch_size=512 --num_gpus=4
python ./eval.py --batch_size=2048 --num_gpus=4 
python ./eval.py --batch_size=8192 --num_gpus=4 
