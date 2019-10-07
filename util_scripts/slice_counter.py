"""
This function counts the number of slices belonging to each class
"""
import numpy as np
import SimpleITK as sitk
import os
import glob
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--folder', nargs=1, help='Name of the folder containing the image data')
parser.add_argument('--view', nargs=1, type=int, help='Which views 0, 1, 2, axial, coronal, sagittal')
parser.add_argument('--n_class', nargs=1, type=int, help="Number of classes")

args = parser.parse_args()

base_name = args.folder[0]
view = args.view[0]
num_class = args.n_class[0]

data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
view_names = ['axial', 'coronal', 'sagittal']

y_names = []
for subject_dir in sorted(glob.glob(os.path.join(data_folder, "ct_train_masks", '*.nii.gz'))):
    y_names.append(os.path.realpath(subject_dir))

print(len(y_names))
counters = np.zeros([len(y_names), num_class+1])
for i, fn in enumerate(y_names):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(fn))
    np.unique(mask)
    counters[i][0] = mask.shape[0]
    for j in range(1,num_class+1):
        ids = np.max(np.max(np.moveaxis(mask,view,0),axis=-1),axis=-1)==j-1
        counters[i][j] = np.sum(ids)
    print(counters[i])
print(counters)
np.save(os.path.join(data_folder, 'counts_'+view_names[view]), counters)
    
