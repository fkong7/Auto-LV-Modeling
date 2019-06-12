# -*- coding: utf-8 -*-

import os
import glob
import functools

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk 
from skimage.transform import resize

from utils import getTrainNLabelNames
from skimage.transform import resize

def transform_func(image, reference_image, transform, order=1):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if order ==1:
      interpolator = sitk.sitkLinear
    elif order == 0:
      interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    try:
      resampled = sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
    except Exception as e: print(e)
      
    return resampled

def reference_image(spacing, size, dim):
 
  reference_size = [256] * dim
  reference_spacing = [int(np.max(size))/256*spacing] * dim
  #reference_size = size
  reference_image = sitk.Image(reference_size, 0)
  reference_image.SetOrigin(np.zeros(3))
  reference_image.SetSpacing(reference_spacing)
  reference_image.SetDirection(np.eye(3).ravel())
  return reference_image

def centering(img, ref_img, order=1):
  dimension = img.GetDimension()
  transform = sitk.AffineTransform(dimension)
  transform.SetMatrix(img.GetDirection())
  transform.SetTranslation(np.array(img.GetOrigin()) - ref_img.GetOrigin())
  # Modify the transformation to align the centers of the original and reference image instead of their origins.
  centering_transform = sitk.TranslationTransform(dimension)
  img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
  reference_center = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
  centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
  centered_transform = sitk.Transform(transform)
  centered_transform.AddTransform(centering_transform)

  return transform_func(img, ref_img, centered_transform, order)

def isometric_transform(image, ref_img, orig_direction, order=1):
  # transform image volume to orientation of eye(dim)
  dim = ref_img.GetDimension()
  affine = sitk.AffineTransform(dim)
  target = np.eye(dim)
  
  ori = np.reshape(orig_direction, np.eye(dim).shape)
  affine.SetMatrix(np.matmul(target,np.linalg.inv(ori)).ravel())
  affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
  #affine.SetMatrix(image.GetDirection())
  return transform_func(image, ref_img, affine, order)

def resample_spacing(sitkIm_fn, resolution=0.5, dim=3, order=1):
  image = sitk.ReadImage(sitkIm_fn)
  orig_direction = image.GetDirection()
  orig_size = np.array(image.GetSize(), dtype=np.int)
  orig_spacing = np.array(image.GetSpacing())
  new_size = orig_size*(orig_spacing/np.array(resolution))
  new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
  new_size = [int(s) for s in new_size]
  
  ref_img = reference_image(resolution, new_size, dim)
  centered = centering(image, ref_img, order)
  transformed = isometric_transform(centered, ref_img, orig_direction, order)
  
  return transformed, ref_img

def resample_scale(sitkIm, ref_img, scale_factor=1., order=1):
  assert type(scale_factor)==np.float64, "Isotropic scaling"
  dim = sitkIm.GetDimension()
  affine = sitk.AffineTransform(dim)
  scale = np.eye(dim)
  np.fill_diagonal(scale, 1./scale_factor)
  
  affine.SetMatrix(scale.ravel())
  affine.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
  transformed = transform_func(sitkIm, ref_img, affine, order)
  return transformed

def cropMask(mask, percentage):
  ori_shape = mask.shape
  print("Original shape before cropping: ", ori_shape)
  # crop the surroundings by percentage
  def boolCounter(boolArr):
    #Count consecutive occurences of values varying in length in a numpy array
    out = np.diff(np.where(np.concatenate(([boolArr[0]],
                                     boolArr[:-1] != boolArr[1:],
                                     [True])))[0])[::2]
    return out
  
  dim  = len(mask.shape)
  for i in range(dim):
    tmp = np.moveaxis(mask, i, 0)
    IDs = np.max(np.max(tmp,axis=-1),axis=-1)==0
    blank = boolCounter(IDs)
    upper = int(blank[0]*percentage) if int(blank[0]*percentage) != 0 else 1
    lower = -1*int(blank[-1]*percentage) if int(blank[-1]*percentage) !=0 else -1
    mask = np.moveaxis(tmp[int(blank[0]*percentage): -1*int(blank[-1]*percentage),:,:],0,i)
    
  print("Final shape post cropping: ", mask.shape)
  ratio = np.array(mask.shape)/np.array(ori_shape)
  return mask, ratio

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

def sample_in_range(range_):
  return (range_[1] - range_[0]) * np.random.random_sample() + range_[0]
  
def main():

    modality = ["ct", "mr"]
    data_folder = '/content/gdrive/My Drive/ImageData/MMWHS'
    output_data_folder = ''
    
    filenames_dic = {}
    for m in modality:
        x_train_filenames, y_train_filenames = getTrainNLabelNames(data_folder, m)
        print("Number of training volumes %d" % len(x_train_filenames_ct))
        print("Number of mask volumes %d" % len(y_train_filenames_ct))
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
    
    

    def _augment(train_filenames, range_adjust, ratio, order=1):
        scale_factor = np.mean([sample_in_range(range_adjust[0]), sample_in_range(range_adjust[1]), sample_in_range(range_adjust[2])]/ratio)
        sitkIm, ref_img = resample_spacing(train_filenames, order=order)
        image = resample_scale(sitkIm, ref_img, scale_factor, order)
        return image
      
    def _writeIm(fn, image):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(fn)
        writer.Execute(image)
        return 
        
    for m in modality:
      num = len(filenames_dic[m+'_x'])
      aug_num = 10
      for i in range(num):
          fn = os.path.join(output_data_folder, os.path.basename(filenames_dic[m+'_x'][i]))
          img = resample_spacing(filenames_dic[m+'_x'][i], order=1)
          _writeIm(fn, img)

          fn = os.path.join(output_data_folder, os.path.basename(filenames_dic[m+'_y'][i]))
          mask = resample_spacing(filenames_dic[m+'_y'][i], order=0)
          _writeIm(fn, mask)
          
          for j in range(aug_num):
              mask = _augment(filenames_dic[m+'_y'][i], range_adjust, ratios[m][i,:], order=0)
              fn =  os.path.join(output_data_folder, m+'_aug_'+str(i)+'_'+str(j)+'_label.nii.gz')
              _writeIm(fn, mask)
              img = _augment(filenames_dic[m+'_x'][i], range_adjust, ratios[m][i,:], order=1)
              fn =  os.path.join(output_data_folder, m+'_aug_'+str(i)+'_'+str(j)+'_image.nii.gz')
              _writeIm(fn, mask)



