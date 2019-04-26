import sys
import os
import numpy as np
import utils
from sklearn.model_selection import KFold
from shutil import copyfile
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder', nargs=1, help='Name of the folder containing the image data')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--n_repeat', nargs='?',const=1, default=1,type=int,help='Number of repeats for cross validation')
parser.add_argument('--n_split', nargs='?', const=5, default=5,type=int, help='Number of splits for cross validation')
parser.add_argument('--n_test', nargs=1, type=int, help='Number of volumes for testing')
args = parser.parse_args()

base_name = args.folder[0]
data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
modality = args.modality
data_out_folder = data_folder + '_CrossValidation'
run_num = args.n_repeat
num_partition = args.n_split
num_test = args.n_test[0]

def copy(fns, data_out_folder):
    for t in fns:
        name = os.path.basename(t)
        newname = os.path.join(data_out_folder, name)
        copyfile(t, newname)
    return

def writeFiles(data_out_folder, modality, img_train,img_val, img_test, mask_train, mask_val, mask_test):
    try:
        os.mkdir(data_out_folder)
    except Exception as e: print(e)

    try:
        os.mkdir(os.path.join(data_out_folder, modality+'_train'))
        os.mkdir(os.path.join(data_out_folder, modality+'_val'))
        os.mkdir(os.path.join(data_out_folder, modality+'_test'))
        os.mkdir(os.path.join(data_out_folder, modality+'_train_masks'))
        os.mkdir(os.path.join(data_out_folder, modality+'_val_masks'))
        os.mkdir(os.path.join(data_out_folder, modality+'_test_masks'))
    except Exception as e: print(e)

    copy(img_train, os.path.join(data_out_folder, modality+'_train'))
    copy(img_val, os.path.join(data_out_folder, modality+'_val'))
    copy(img_test, os.path.join(data_out_folder, modality+'_test'))
    copy(mask_train, os.path.join(data_out_folder, modality+'_train_masks'))
    copy(mask_val, os.path.join(data_out_folder, modality+'_val_masks'))
    copy(mask_test, os.path.join(data_out_folder, modality+'_test_masks'))
    
    return

def partition(data_folder, data_out_folder, modality, num_partition, n):
    for m in modality:
        imgVol_fn, mask_fn = utils.getTrainNLabelNames(data_folder,m)
        
        indices = np.array(range(len(imgVol_fn)))
        #leave out the last n for testing
        indices = indices[:-n]
    
        kf = KFold(n_splits=num_partition, shuffle=True)
        fold = 0
        for train_index, test_index in kf.split(indices):
            x_train, x_test = indices[train_index], indices[test_index]
            x_train = x_train.astype(int)
            x_test = x_test.astype(int)
            imgVol_fn_train = [imgVol_fn[i] for i in x_train]
            imgVol_fn_val = [imgVol_fn[i] for i in x_test]
            mask_fn_train = [mask_fn[i] for i in x_train]
            mask_fn_val = [mask_fn[i] for i in x_test]
            writeFiles(os.path.join(data_out_folder,'fold'+str(fold)), m, imgVol_fn_train, imgVol_fn_val, imgVol_fn[-n:], mask_fn_train, mask_fn_val, mask_fn[-n:])
            fold +=1

def organize(data_folder):
    #function for Kits19 data only to arrange the folders into image and mask folders
    img_dir=os.path.join(data_folder, 'ct_train')
    mask_dir = os.path.join(data_folder, 'ct_train_masks')
    try:
        os.mkdir(img_dir)
        os.mkdir(mask_dir)
    except Exception as e: print(e)
    import glob
    for subject_dir in sorted(glob.glob(os.path.join(data_folder,'case*'))):
        if os.path.isdir(subject_dir):
            new_name = os.path.basename(subject_dir)+'img.nii.gz'
            new_mask_name = os.path.basename(subject_dir)+'label.nii.gz'
            old_name = os.path.join(subject_dir,'imaging.nii.gz')
            old_mask_name = os.path.join(subject_dir,'segmentation.nii.gz')

            copyfile(old_name, os.path.join(img_dir,new_name))
            copyfile(old_mask_name, os.path.join(mask_dir, new_mask_name))

organize(data_folder)
try:
    os.mkdir(data_out_folder)
except Exception as e: print(e)

for i in range(run_num):
    subfolder = os.path.join(data_out_folder, 'run'+str(i))
    try:
        os.mkdir(subfolder)
    except Exception as e: print(e)
    partition(data_folder, subfolder, modality, num_partition,num_test)
