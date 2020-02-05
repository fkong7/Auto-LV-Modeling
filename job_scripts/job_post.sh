#!/bin/bash
# Job name:
#SBATCH --job-name=2dunet
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=24
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=10:30:00
#
## Command(s) to run (example):
module load gcc openmpi python

mpirun -np 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/util_scripts/post_processing.py \
    --folder 2DUNet/Logs/MMWHS_aug2/run1/test_ensemble \
    --output 2DUNet/Logs/MMWHS_aug2/run1/test_ensemble_post \
