sv_python_dir=/Users/fanweikong/SimVascular/build/SimVascular-build

json_file=/Users/fanweikong/Documents/Modeling/SurfaceModeling/info2.json

model_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/main.py
registration_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/elastix_main.py
volume_mesh_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/volume_mesh_main.py

#${sv_python_dir}/sv --python -- ${model_script} --json_fn ${json_file}

#dir=/Users/fanweikong/Downloads/test_ensemble-2-10-2
#dir=/Users/fanweikong/Documents/Modeling/SurfaceModeling/examples
#dir=/Users/fanweikong/Documents/ImageData/orCalScore_CTAI/ct_train_masks
#dir=/Users/fanweikong/Documents/ImageData/4DCCTA/MACS40244_20150309/wall_motion_labels_gt
#dir=/Users/fanweikong/Documents/ImageData/MMWHS/test_ensemble-2-10-2_seg_corrected
dir=/Users/fanweikong/Documents/ImageData/MMWHS/temp
ls ${dir}
for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/sv --python -- ${model_script} --json_fn ${json_file} --seg_name ${file##*/}; done

#conda activate elastix
#python ${registration_script} --json_fn ${json_file} --write --smooth
#conda deactivate

#${sv_python_dir}/sv --python -- ${volume_mesh_script} --json_fn ${json_file}

