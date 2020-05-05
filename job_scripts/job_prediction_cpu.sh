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
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=04:30:00
#
## Command(s) to run (example):
#module load gcc openmpi python
module load python/3.6
module load tensorflow/1.12.0-py36-pip-gpu
module load cuda

python /global/scratch/fanwei_kong/DeepLearning/2DUNet/prediction.py \
    --image ImageData/MMWHS_small \
    --output 2DUNet/Logs/MMWHS_editted_aug/run0 \
    --model 2DUNet/Logs/MMWHS_editted_aug/run0\
    --view 0 1 2 \
    --modality ct mr \
    --mode validate

