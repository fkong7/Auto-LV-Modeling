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
module load tensorflow

python /global/scratch/fanwei_kong/DeepLearning/2DUNet/util_scripts/data_alignment.py \
    --image_dir ImageData/MMWHS_editted \
    --output ImageData/MMWHS_editted_aug \
    --modality ct mr \
    --folder_attr _train\
    --aug_num 10 \
#    --intensity

python /global/scratch/fanwei_kong/DeepLearning/2DUNet/util_scripts/data_alignment.py \
    --image_dir ImageData/MMWHS_editted \
    --output ImageData/MMWHS_editted_aug \
    --modality ct mr \
    --folder_attr _val\
    --aug_num 0

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/util_scripts/data_alignment.py \
#    --image_dir ImageData/MMWHS \
#    --output ImageData/MMWHS_aug4 \
#    --modality ct mr \
#    --folder_attr _val\
#    --aug_num 0
