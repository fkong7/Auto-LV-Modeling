# Use SimVascular to construct LV volme mesh

# Path to SimVascular exectuable
sv_python_dir=/Applications/SimVascular.app/Contents/Resources

volume_mesh_script=Modeling/volume_mesh_main.py

# Path to the surface meshes
input_dir=./04-SurfReg/BD9702
# Path to the outputed volume meshes
output_dir=./05-VolMesh/BD9702
# Name format of surface
surf_format=frame%03d.nii.gz.vtp

# Volumetric Meshing using SimVascular
${sv_python_dir}/simvascular --python \
    -- ${volume_mesh_script} \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_out $surf_format \
    --edge_size 2.5

