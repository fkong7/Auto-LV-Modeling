sv_python_dir=/usr/local/bin

json_file=Modeling/info.json

model_script=Modeling/main.py
registration_script=Modeling/elastix_main.py
volume_mesh_script=Modeling/volume_mesh_main.py

dir=./examples/ct_test_seg
output_dir=./output

ls ${dir}
for file in ${dir}/*.nii.gz; do echo ${file} &&  ${sv_python_dir}/simvascular --python -- ${model_script} --input_dir ${dir} --output_dir ${output_dir} --seg_name ${file##*/} --edge_size 2.5; done

#for file in ${dir}/*.nii.gz; do echo ${file} &&  python ${model_script} --json_fn ${json_file} --seg_name ${file##*/} --disable_SV; done

