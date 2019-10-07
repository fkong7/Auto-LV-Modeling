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
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
module load python
module load tensorflow/1.12.0-py36-pip-gpu
module load cuda
python /global/scratch/fanwei_kong/DeepLearning/2DUNet/2dunet_multiclass.py MMWHS MMWHS/total_run3 41 8 500 1 2
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py MMWHS_CrossValidation/run0/fold0 MMWHS_CrossValidation/run0/fold0_0 41
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py MMWHS_CrossValidation/run0/fold0 MMWHS_CrossValidation/run0/fold0_1 21
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py MMWHS_CrossValidation/run0/fold0 MMWHS_CrossValidation/run0/fold0_2 11


