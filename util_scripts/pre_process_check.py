import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import numpy as np
import SimpleITK as sitk
from preProcess import resample_spacing
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--folder',  help='Name of the folder containing the image data')
parser.add_argument('--out_folder', help='Folder postfix of the folder to look for')
parser.add_argument('--intensity',nargs='+', type=int, default=[750,-750], help='Intensity range to clip to [upper, lower]')
args = parser.parse_args()

fns = glob.glob(os.path.join(args.folder, '*.nii.gz'))+glob.glob(os.path.join(args.folder, '*.nii'))

try:
    os.makedirs(os.path.join(args.out_folder))
except:
    pass

spacing = []
spacing_ori = []
for fn in fns:
    img = sitk.ReadImage(fn)
    spacing_ori.append(img.GetSpacing())
    img = resample_spacing(img, order=1)[0]
    spacing.append(img.GetSpacing())
    fn_out = os.path.join(args.out_folder, os.path.basename(fn))
    sitk.WriteImage(img, fn_out)
spacing = np.array(spacing)
spacing_ori = np.array(spacing_ori)

print("Axial: ", np.mean(spacing[:,0]), np.min(spacing[:,0]), np.max(spacing[:,0]))
print("Sagittal: ", np.mean(spacing[:,1]), np.min(spacing[:,1]), np.max(spacing[:,1]))
print("Coronal: ", np.mean(spacing[:,2]), np.min(spacing[:,2]), np.max(spacing[:,2]))
print("Axial: ", np.mean(spacing_ori[:,0]), np.min(spacing_ori[:,0]), np.max(spacing_ori[:,0]))
print("Sagittal: ", np.mean(spacing_ori[:,1]), np.min(spacing_ori[:,1]), np.max(spacing_ori[:,1]))
print("Coronal: ", np.mean(spacing_ori[:,2]), np.min(spacing_ori[:,2]), np.max(spacing_ori[:,2]))

np.save(os.path.join(args.out_folder, 'spacing.npy'), spacing)
