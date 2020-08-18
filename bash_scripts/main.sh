#sv_python_dir=/Users/fanweikong/SimVascular/build/SimVascular-build
sv_python_dir=/usr/local/bin

json_file=/Users/fanweikong/Documents/Modeling/SurfaceModeling/info2.json

model_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/main.py
registration_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/elastix_main.py
volume_mesh_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/volume_mesh_main.py

dir=./examples
ls ${dir}
for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/simvascular --python -- ${model_script} --json_fn ${json_file} --seg_name ${file##*/}; done

conda activate elastix
python ${registration_script} --json_fn ${json_file} --write --smooth
conda deactivate

${sv_python_dir}/sv --python -- ${volume_mesh_script} --json_fn ${json_file}

