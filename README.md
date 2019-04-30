# 2DUnet

This project works on segmentation of 2D medical image volumes. 

Two methods of segmenting 3D structures are supported:

* 2D segmentation along each axis of 3D volume and then avergae the probability map
* 2.5D segmentation along the sagittal axis of the 3D volume (multi-channel)

## Dependencies

* Tensorflow
* Python 
* SimpleItk

## Partition Dataset 

We use cross-validation to find the optimal hyperparameters. To partition the data, on the login node, run:

```
python partition.py --folder <image_folder> --modality ct --n_repeat 1 --n_split 5 --n_test 35
```
So for the Kits19 data, we are using a 5 fold split while leaving out 35 (out of 210) volumes as testing data

## Image Preprocesing

The image preprocessing is done before training. 3D image volumes are processed and sliced into 2D images or 3d sub-volumes if using 2.5D segmentation. 

```
sbatch job_cpu.sh
```
The command in the job script looks like:
```
mpirun -n 24 python /global/scratch/fanwei_kong/2DUNet/image2tfrecords.py --folder Kits19_CrossValidation/run0/fold0 --view 0 --modality ct --out_folder _train --n_channel 1 --intensity 512 -512
```
where `--n_channel` is the number of channels for 2.5D segmentation; `--view` is the view id (axial=0, coronal=1, sagittal=2)

## Training

To train the model, submit the following script:
```
sbatch job_sagittal.sh
```
The command in the job script looks like:
```
python /global/scratch/fanwei_kong/2DUNet/2dunet_multiclass_sagittal.py <image_folder> <save_folder> <random_seed> <view_id> <n_epoch>
```

## Prediction
To generate prediction and dice scores, submit the following script:
```
sbatch job_prediction.sh
```

TO-DO: Edit `2dunet_prediction.py` to use argparser

