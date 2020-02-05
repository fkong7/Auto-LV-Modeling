python ~/2DUNet/prediction.py \
        --image ~/Documents/ImageData/MMWHS \
        --output 2DUNet/Logs/MMWHS_aug2/run1/test_ensemble \
        --model 2DUNet/Logs/MMWHS_aug2/run1 \
        --view 0 1 2 \
        --modality ct mr \
        --mode test
