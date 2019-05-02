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
module load tensorflow/1.12.0-py36-pip-gpu
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 0 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 0 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 1 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 1 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 2 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold1 --view 2 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512

mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 0 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 0 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 1 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 1 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 2 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold2 --view 2 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512

mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 0 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 0 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 1 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 1 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 2 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold3 --view 2 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512

mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 0 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 0 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 1 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 1 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 2 --modality ct --out_folder _train --n_channel 3 --intensity 512 -512
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold4 --view 2 --modality ct --out_folder _val --n_channel 3 --intensity 512 -512
