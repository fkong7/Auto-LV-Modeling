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
module load python
module load tensorflow/1.12.0-py36-pip-gpu
module load cuda

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug3 \
#    --attr 1 \
#    --output MMWHS_aug3/total_run0 \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug3 \
#    --attr 1 \
#    --output MMWHS_aug3/total_run1 \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.005

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run0 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.005
#CHANGED Patience from 3 to 10 (lr schedule)
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run1 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.005
# CHANGE LR Scale factor from 0.5 to 0.8, tf clip on intensity -1 to 1
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run2 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.005
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run3 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.002
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run4 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.001

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run5 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.0005
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run6 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.0001

#change min lr to 0.00005
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/mr_run7 \
#    --view 2 \
#    --modality mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.00005

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_3 \
#    --attr 1 \
#    --output MMWHS_3/total_run0 \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.0001

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug4 \
#    --attr 1 \
#    --output MMWHS_aug4/total_run0 \
#    --view 2 \
#    --modality ct mr \
##    --num_epoch 500 \
##    --num_class 8 \
#    --lr 0.0001
#
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug4 \
#    --attr 1 \
#    --output MMWHS_aug4/total_run1 \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.00005 \
#    --batch_size 24

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug4_intensity \
#    --attr 1 \
#    --output MMWHS_aug4_intensity/total_run0 \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.00005 \
#    --batch_size 24

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py \
#    --image MMWHS_aug2 \
#    --attr 1 \
#    --output MMWHS_aug2/total_run_small_lr \
#    --view 2 \
#    --modality ct mr \
#    --num_epoch 500 \
#    --num_class 8 \
#    --lr 0.00005 \
#    --batch_size 24


# CHANGED TO MEAN_DICE_COEFF
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
