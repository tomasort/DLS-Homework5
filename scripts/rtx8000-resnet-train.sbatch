#!/bin/bash
#SBATCH --job-name=rtx8000-tta-experiment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:rtx8000:1

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS-Homework5/

python train_resnet50.py --dataset /scratch/tor213/data --gpu_name rtx8000