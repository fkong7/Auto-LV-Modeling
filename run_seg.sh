image_dir=examples
output_dir=output
weight_dir=/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble_4_20_weights

python ./Segmentation/prediction.py \
    --image $image_dir \
    --output $output_dir \
    --model $weight_dir \
    --view 0 \
    --modality ct 
