sv_python_dir=/Users/fanweikong/SimVascular/build/SimVascular-build

json_file=/Users/fanweikong/Documents/Modeling/SurfaceModeling/info.json

model_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/main.py
registration_script=/Users/fanweikong/Documents/Modeling/SurfaceModeling/elastix_main.py

#${sv_python_dir}/sv --python -- ${model_script} --json_fn ${json_file}


conda activate elastix
python ${registration_script} --json_fn ${json_file} --write
conda deactivate

