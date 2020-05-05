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
#SBATCH --time=30:30:00
#
## Command(s) to run (example):
module load gcc/5.4.0
module load python/3.5
module load boost/1.63.0-gcc
module load hdf5/1.8.18-gcc-p
module load openmpi/2.0.2-gcc
module load netcdf/4.4.1.1-gcc-p
module load cmake/3.7.2
module load swig/3.0.12
#module load tensorflow/1.12.0-py36-pip-gpu
module load tensorflow/1.7.0-py35-pip-cpu

mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 0 --modality ct mr --out_folder _train --n_channel 1 --attr 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 0 --modality ct mr --out_folder _val --n_channel 1 --attr 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 1 --modality ct mr --out_folder _train --n_channel 1 --attr 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 1 --modality ct mr --out_folder _val --n_channel 1 --attr 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 2 --modality ct mr --out_folder _train --n_channel 1 --attr 1
mpirun -n 24 python /global/scratch/fanwei_kong/DeepLearning/2DUNet/image2tfrecords.py --folder MMWHS_editted_aug --view 2 --modality ct mr --out_folder _val --n_channel 1 --attr 1
