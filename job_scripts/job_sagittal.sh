#!/bin/bash
# Job name:
#SBATCH --job-name=2dunet
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
## Command(s) to run (example):
module load python/3.6
module load tensorflow/1.12.0-py36-pip-gpu
module load cuda

python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
    --image MMWHS_editted_aug \
    --attr 1 \
    --output MMWHS_editted_aug/run2 \
    --view 2 \
    --modality ct mr \
    --num_epoch 500 \
    --num_class 8 \
    --lr 0.0005 \
    --batch_size 24
