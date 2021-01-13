# Use SimpleElastix to register surface meshes

# Path to the ct/mr images or segmentation results
image_dir=./01-Images/ct_test/BD9702
# Path to the mask file
mask_dir=./02-Segmnts/BD9702
# Path to the unregistered surface mesh
surface_dir=./03-Surfaces/BD9702
# Path to the registered surface meshes
output_dir=./04-SurfReg/BD9702

# Phase ID of the surface mesh used as the registration target
start_phase=8
# Total number of phases
#total_phase=10

# Input format of unregistered surfaces
surf_format=frame%03d.nii.gz.vtp
# Name of the images in image_dir
imag_format=frame%03d.nii.gz

# Registration with SimpleElastix
python Modeling/elastix_main.py \
    --image_dir $mask_dir \
    --mask_dir $mask_dir \
    --surface_dir $surface_dir \
    --output_dir $output_dir \
    --start_phase $start_phase \
    --model_output $surf_format \
    --im_name $imag_format \
    --edge_size 2.5 \
    --write \
    --smooth
