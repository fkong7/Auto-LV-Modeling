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
#module load tensorflow/1.12.0-py36-pip-gpu
module load tensorflow

#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/prediction_ensemble.py
#python /global/scratch/fanwei_kong/DeepLearning/2DUNet/prediction.py
#python /global/scratch/fanwei_kong/2DUNet/partition.py
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS --view 0 --modality ct mr --out_folder _train --n_channel 1
#mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_2 --view 0 --modality ct mr --out_folder _val --n_channel 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS --view 1 --modality ct mr --out_folder _train --n_channel 1
#mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_2 --view 1 --modality ct mr --out_folder _val --n_channel 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS --view 2 --modality ct mr --out_folder _train --n_channel 1
#mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_2 --view 2 --modality ct mr --out_folder _val --n_channel 1
