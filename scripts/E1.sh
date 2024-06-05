# !/bin/bash
# SBATCH --job-name=E1
# SBATCH --gres=gpu:rtx8000:1
# SBATCH --cpus-per-task=1
# SBATCH --output=%x.result
# SBATCH --mem=10GB
# SBATCH --time=01:00:00

conda activate resnet-eval

# E1: I/O optimization
python ./eval.py --batch_size=32 --num_workers=0
python ./eval.py --batch_size=32 --num_workers=2
python ./eval.py --batch_size=32 --num_workers=4
python ./eval.py --batch_size=32 --num_workers=8
python ./eval.py --batch_size=32 --num_workers=12
python ./eval.py --batch_size=32 --num_workers=16