# -*- coding: utf-8 -*-

import os
import glob
import functools

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk 
from skimage.transform import resize

from utils import getTrainNLabelNames, writeIm, sample_in_range
from skimage.transform import resize
from preProcess import resample_scale, resample_spacing, cropMask

def blankSpaces(y_train_filenames_ct, y_train_filenames_mr):
  num = len(y_train_filenames_ct)
  ratio_mr = []
  ratio_ct = []
  for i in range(num):
    mask_ct, ratio = cropMask(sitk.GetArrayFromImage(resample_spacing(y_train_filenames_ct[i])[0]),1.)
    print(ratio)
    ratio_ct = np.concatenate((ratio_ct, ratio))
    mask_mr, ratio = cropMask(sitk.GetArrayFromImage(resample_spacing(y_train_filenames_mr[i])[0]),1.)
    print(ratio)
    ratio_mr = np.concatenate((ratio_mr, ratio))
  return ratio_ct, ratio_mr

  
def main():

    modality = ["ct", "mr"]
    data_folder = '/Users/fanweikong/Documents/ImageData/MMWHS_small/'
    fdr_postfix = '_train'
    output_data_folder = '/Users/fanweikong/Documents/ImageData/MMWHS_small_aug'
    if fdr_postfix == '_train':
        ids = range(0,12)
    elif fdr_postfix =='_val':
        ids = range(12,16)

    for m in modality:
        try:
            os.makedirs(os.path.join(output_data_folder,m+fdr_postfix))
        except Exception as e: print(e)
        try:
            os.makedirs(os.path.join(output_data_folder,m+fdr_postfix+'_masks'))
        except Exception as e: print(e)
    
    filenames_dic = {}
    for m in modality:
        x_train_filenames, y_train_filenames = getTrainNLabelNames(data_folder, m)
        print("Number of training volumes %d" % len(x_train_filenames))
        print("Number of mask volumes %d" % len(y_train_filenames))
        filenames_dic[m+'_x'] = x_train_filenames
        filenames_dic[m+'_y'] = y_train_filenames

    #find the blank spaces for image volumes 
    ratios = {}
    ratios['ct'], ratios['mr'] = blankSpaces(filenames_dic['ct_y'], filenames_dic['mr_y'])

    for m in modality:
    	ratios[m] = ratios[m].reshape(int(len(ratios[m])/3),3)
    #assume uniform distribution, find the range of blank space ratios
    dim = 3
    range_adjust = [None]*dim
    for i in range(dim):
        range_adjust[i] = [np.min([np.min(ratios[m][:,i]) for m in modality]), np.max([np.max(ratios[m][:,i]) for m in modality])]
    print("The ratio of blank space found for ct and mr is: ", range_adjust)



    # Apply data augmentation so that the scale, spacing, orientation of MR and CT volumes are consistant
    
    

    def _augment(train_filenames, range_adjust, ratio, scale_factor=None, order=1):
        if scale_factor is None:
            scale_factor = np.mean([sample_in_range(range_adjust[0]), sample_in_range(range_adjust[1]), sample_in_range(range_adjust[2])]/ratio)
        sitkIm, ref_img = resample_spacing(train_filenames, order=order)
        image = resample_scale(sitkIm, ref_img, scale_factor, order)
        return image, scale_factor
      
        
    for m in modality:
      num = len(filenames_dic[m+'_x'])
      aug_num = 10
      for i in ids:
          fn = os.path.join(output_data_folder, m+fdr_postfix, os.path.basename(filenames_dic[m+'_x'][i]))
          img,_ = resample_spacing(filenames_dic[m+'_x'][i], order=1)
          _writeIm(fn, img)

          fn = os.path.join(output_data_folder, m+fdr_postfix+'_masks', os.path.basename(filenames_dic[m+'_y'][i]))
          mask,_ = resample_spacing(filenames_dic[m+'_y'][i], order=0)
          _writeIm(fn, mask)
          
          for j in range(aug_num):
              mask, scale_factor = _augment(filenames_dic[m+'_y'][i], range_adjust, ratios[m][i,:], order=0)
              fn =  os.path.join(output_data_folder, m+fdr_postfix+'_masks', m+'_aug_'+str(i)+'_'+str(j)+'_label.nii.gz')
              _writeIm(fn, mask)
              img, _ = _augment(filenames_dic[m+'_x'][i], range_adjust, ratios[m][i,:], scale_factor=scale_factor, order=1)
              fn =  os.path.join(output_data_folder, m+fdr_postfix, m+'_aug_'+str(i)+'_'+str(j)+'_image.nii.gz')
              _writeIm(fn, img)


if __name__ == '__main__':
    main()
