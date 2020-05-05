# Ensemble of 2D UNets for automatic segmentation of 3D medical image volumes

This project implemented UNet for automatic segmentation of 3D medical image volumes. 

Since CNN-based 3D segmentation algorithms usually require high memory consumption and computational cost, while 2D CNN-based algorithms ignore spatial connection between adjacent slices, we utilized an ensemble of 2D CNNs to generate 3D segmentation at high resolution (256x256x256). We sliced the 3D image volumes along the axial, sagittal or coronalaxis to obtain corresponding 2D image datasets. The 2D predictions fromeach CNN model were stacked together to form 3D predictions that were then averaged across allmodels. 


Two methods of slicing and segmenting the 3D image volumes are supported:

* 2D segmentation along each axis of 3D volume 
* 2.5D segmentation along each axis of the 3D volume (multi-channel)
## Training data 
We used the image and ground truth data provided by [MMWHS] (http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) to train our models. 
Our segmentation models were trainined simultaneously on CT and MR data and trained weights are [here] (https://drive.google.com/open?id=162Xr5OezSZL-0K3aoYO7WnHWuGTEXkkj). 
## Dependencies

* Tensorflow
* Python 
* SimpleITK

## Image Preprocesing

The image preprocessing was appliedbefore training. 3D image volumes are processed and sliced into 2D images or 3d sub-volumes if using 2.5D segmentation. 

```
sbatch job_cpu.sh
```
The command in the job script looks like:
```
mpirun -n 24 python image2tfrecords.py --folder /path/to/image/folder --view 0 --modality ct --out_folder _train --n_channel 1 --intensity 512 -512
```
where `--n_channel` is the number of channels for 2.5D segmentation; `--view` is the view id (axial=0, coronal=1, sagittal=2)

## Training

To train the model, submit the following script:
```
sbatch job_sagittal.sh
```
## Prediction
To generate prediction and dice scores, submit the following script:
```
sbatch job_prediction.sh
```
