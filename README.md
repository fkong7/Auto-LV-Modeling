# Automating Model Generation For Image-Based Cardiac Flow Simulation

This repository contains the source code for our paper:

Kong, F., and Shadden, S. C. (August 7, 2020). "Automating Model Generation for Imagebased Cardiac Flow Simulation." ASME. J Biomech Eng. doi: https://doi.org/10.1115/1.4048032

The code repository consists of two parts

* Deep-learning based automatic segmentation of 3D CT/MR image data using an ensemble of 2D UNets
* Down-stream automatic model generation from segmentations for LV CFD simulations 

## Dependencies

 * [SimVascular](https://github.com/SimVascular/SimVascular) 

## Segmentation Usage 

The segmentation models can generate segmentations for LV blood pool, LV myocardium, LA, RA, RV blood pool, aorta and pulmonary artery.
### Input Requirements
The preferred input format of the image volumes is **.nii.gz or nii**. VTK image volumes (.vti) are also accepted; however they should be reoriented to have an orientation matrix of identity. This is because the segmnetation method requires identity-oriented image volumes while the version of VTK within SimVascular does not include orientation matrix with VTI images. 
The directory containing the input image data should be organized as follows:
```
image_data_dir
    |__ ct_test
          |__ image_volume1.nii.gz
          |__ image_volume2.nii.gz
          |__ image_volume3.nii.gz
```
### Trained Models
We used the image and ground truth data provided by [MMWHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) to train our models. 
Our segmentation models were trainined simultaneously on CT and MR data and trained weights are [here](https://drive.google.com/open?id=162Xr5OezSZL-0K3aoYO7WnHWuGTEXkkj). 

### Prediction
To generate segmentations for 3D CT or MR image volumes:
```
sv_python_dir=/usr/local/bin
${sv_python_dir}/simvascular  --python -- Segmentation/prediction.py \
    --image /path/to/image/dir \ # the images should be saved in nii.gz format under a folder named as [modality]_test within /path/to/image/dir. 
    --output /path/to/output \
    --model /path/to/model/weights \
    --view 0 1 2 \ # Use models trained on axial (0), coronal (1) and/or sagittal (2) view[s].
    --modality ct \ # Image modality, ct or mr
    --mode test
```

## LV Modeling Usage

The model construction pipeline takes in the generated segmentation and output reconstructed LV surface meshes for CFD simulations. 

### Construct LV Surface Meshes with Tagged Boundary Faces
* Update `info.json` with correct file and folder names.
* Run `main.py` to generate a LV surface mesh for each segmentation file in a folder.   
    ```
    sv_python_dir=/usr/local/bin
    model_script=Modeling/main.py
    dir=./examples/ct_test_seg
    for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/simvascular --python -- ${model_script} --input_dir ${dir} --output_dir ${output_dir} --seg_name ${file##*/} --edge_size 2.5; done
    ```
* Use `--disable_SV` to turn off SimVascular (no remeshing would be performed). 
    ```
    for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/simvascular --python -- ${model_script} --input_dir ${dir} --output_dir ${output_dir} --seg_name ${file##*/} --edge_size 2.5 --disable_SV; done
    ```
    
## Acknowledgement
This work was supported by the NSF, Award #1663747. 

