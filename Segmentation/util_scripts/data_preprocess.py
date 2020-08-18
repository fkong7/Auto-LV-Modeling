import os
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
import SimpleITK as sitk
from utils import np_to_tfrecords
from utils import getTrainNLabelNames
from preProcess import swapLabels, RescaleIntensity, HistogramEqualization, resample_spacing, centering
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Name of the folder containing the image data')
parser.add_argument('--modality', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--out_folder', help='Output folder')
parser.add_argument('--intensity',nargs='+', type=int, default=[750,-750], help='Intensity range to clip to [upper, lower]')
args = parser.parse_args()

imgVol = sorted(glob.glob(os.path.join(args.folder, "*.nii.gz")))
print(imgVol)
try:
    os.makedirs(args.out_folder)
except Exception as e: print(e)

def resample_prediction(im, orig_im):
        #resample prediction so it matches the original image
        im_info = resample_spacing(orig_im, order=1)[0]
        im.SetSpacing(im_info.GetSpacing())
        im.SetOrigin(im_info.GetOrigin())
        im.SetDirection(im_info.GetDirection())
        return centering(im, orig_im, order=1)

for im_fn in imgVol:
    print(im_fn)
    im_ori = resample_spacing(im_fn, order=1)[0]
    im = sitk.GetArrayFromImage(im_ori)
    im = RescaleIntensity(im, args.modality, args.intensity)

    print(np.max(im), np.min(im), im.shape, im_ori.GetSpacing())
    out_im_fn = os.path.join(args.out_folder, os.path.basename(im_fn))
    im = sitk.GetImageFromArray(im)
    im_2 = resample_prediction(im, sitk.ReadImage(im_fn))
    sitk.WriteImage(im, out_im_fn)
    sitk.WriteImage(im_2, out_im_fn)
