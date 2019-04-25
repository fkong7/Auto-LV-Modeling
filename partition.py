import sys
import os
import numpy as np
from utils import getTrainNLabelNames
from sklearn.model_selection import KFold
from shutil import copyfile


base_name = 'MMWHS'
data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
modality = ["ct","mr"]
data_out_folder = data_folder + '_CrossValidation'

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

def partition(data_folder, data_out_folder, modality):
    for m in modality:
        imgVol_fn, mask_fn = getTrainNLabelNames(data_folder,m)
        indices = np.array(range(len(imgVol_fn)))
        #leave out the last four for testing
        indices = indices[:-4]
    
        kf = KFold(n_splits=4, shuffle=True)
        fold = 0
        for train_index, test_index in kf.split(indices):
            x_train, x_test = indices[train_index], indices[test_index]
            x_train = x_train.astype(int)
            x_test = x_test.astype(int)
            imgVol_fn_train = [imgVol_fn[i] for i in x_train]
            imgVol_fn_val = [imgVol_fn[i] for i in x_test]
            mask_fn_train = [mask_fn[i] for i in x_train]
            mask_fn_val = [mask_fn[i] for i in x_test]
            writeFiles(os.path.join(data_out_folder,'fold'+str(fold)), m, imgVol_fn_train, imgVol_fn_val, imgVol_fn[-4:], mask_fn_train, mask_fn_val, mask_fn[-4:])
            fold +=1

try:
    os.mkdir(data_out_folder)
except Exception as e: print(e)

for i in range(3):
    subfolder = os.path.join(data_out_folder, 'run'+str(i))
    try:
        os.mkdir(subfolder)
    except Exception as e: print(e)
    partition(data_folder, subfolder, modality)

