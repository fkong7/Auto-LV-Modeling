# Use SimVascular to construct LV surface meshes

# Path to SimVascular exectuable
sv_python_dir=/usr/local/bin
model_script=Modeling/surface_main.py
# Path to the segmentation results
input_dir=./02-Segmnts/WS01
# Path to the outputed surface meshes
output_dir=./03-Surfaces/WS01

# Construct LV surface meshes with tagged boundary faces
for file in ${input_dir}/*.nii.gz
do 
    echo ${file}
    ${sv_python_dir}/simvascular --python \
        -- ${model_script} \
        --input_dir ${input_dir} \
        --output_dir ${output_dir} \
        --seg_name ${file##*/} \
        --edge_size 3.5
done
