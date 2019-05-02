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
python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold0 Kits19_CrossValidation/run0/fold0_0_multi3 41 3 60 3
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold0 Kits19_CrossValidation/run0/fold0_1 21 3 60
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold0 Kits19_CrossValidation/run0/fold0_2 11 3 60

python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold1 Kits19_CrossValidation/run0/fold1_0_multi3 41 3 60 3
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold1 Kits19_CrossValidation/run0/fold1_1 21 3 60
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold1 Kits19_CrossValidation/run0/fold1_2 11 3 60

python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold2 Kits19_CrossValidation/run0/fold2_0_multi3 41 3 60 3
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold2 Kits19_CrossValidation/run0/fold2_1 21 3 60
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold2 Kits19_CrossValidation/run0/fold2_2 11 3 60

python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold3 Kits19_CrossValidation/run0/fold3_0_multi3 41 3 60 3
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold3 Kits19_CrossValidation/run0/fold3_1 21 3 60
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold3 Kits19_CrossValidation/run0/fold3_2 11 3 60

python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold4 Kits19_CrossValidation/run0/fold4_0_multi3 41 3 60 3
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold4 Kits19_CrossValidation/run0/fold4_1 21 3 60
#python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py Kits19_CrossValidation/run0/fold4 Kits19_CrossValidation/run0/fold4_2 11 3 60

