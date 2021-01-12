patient_id=BD9702
image_dir=01-Images
output_dir=02-Segmnts
weight_dir=./Weights

python ./Segmentation/prediction.py \
    --pid $patient_id \
    --image $image_dir \
    --output $output_dir \
    --model $weight_dir \
    --view  0 1 2 \
    --modality ct
