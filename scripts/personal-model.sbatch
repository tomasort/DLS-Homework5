#!/bin/bash
#SBATCH --job-name=personal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS-Homework5/

bash train-car-person-model.bash
