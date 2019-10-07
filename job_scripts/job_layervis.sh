#!/bin/bash
# Job name:
#SBATCH --job-name=2dunet
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio_bigmem
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=20
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s) to run (example):
module load gcc openmpi python
module load tensorflow/1.10.0-py36-pip-cpu
#python /global/scratch/fanwei_kong/2DUNet/partition.py
mpirun -n 20 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/layer_visualization.py 
