sv_python_dir=/usr/local/bin

json_file=info.json

model_script=main.py
registration_script=elastix_main.py
volume_mesh_script=volume_mesh_main.py

dir=./examples/ct_test_seg
ls ${dir}
#for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/simvascular --python -- ${model_script} --json_fn ${json_file} --seg_name ${file##*/}; done

for file in ${dir}/*.nii.gz; do echo ${file} &&  python ${model_script} --json_fn ${json_file} --seg_name ${file##*/} --disable_SV; done

#conda activate elastix
#python ${registration_script} --json_fn ${json_file} --write --smooth
#conda deactivate
#
#${sv_python_dir}/sv --python -- ${volume_mesh_script} --json_fn ${json_file}
