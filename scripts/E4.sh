#!/bin/bash
#SBATCH --job-name=E4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.result
#SBATCH --mem=10GB
#SBATCH --time=01:00:00

conda activate resnet-eval

# E4: Experimenting with different optimizers
python ./eval.py --batch_size=32 --optimizer=sgd
python ./eval.py --batch_size=32 --optimizer=nesterov
python ./eval.py --batch_size=32 --optimizer=adagrad
python ./eval.py --batch_size=32 --optimizer=adadelta
python ./eval.py --batch_size=32 --optimizer=adam
