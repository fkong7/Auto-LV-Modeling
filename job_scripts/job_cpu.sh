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
#SBATCH --qos=savio_debug
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=12
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:30:00
#
## Command(s) to run (example):
module load gcc openmpi python
module load tensorflow/1.12.0-py36-pip-gpu
#python /global/scratch/fanwei_kong/2DUNet/partition.py
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 0 _train 
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 0 _val
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 1 _train 
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 1 _val 
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 2 _train 
mpirun -n 12 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py MMWHS_CrossValidation/run0/fold3 2 _val 
