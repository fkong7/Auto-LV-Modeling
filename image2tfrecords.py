import os
import glob
import sys

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from mpi4py import MPI
from utils import np_to_tfrecords
from utils import getTrainNLabelNames
from preProcess import swapLabels, RescaleIntensity

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total = comm.Get_size()
"""# Set up"""
#TO-DO:allow arbitrary number of cpus, right now only divisible number of cpus are allowed
modality = ["ct","mr"]
#base_name = 'MMWHS_small_13'
#view = 0 
base_name = str(sys.argv[1])
view = int(sys.argv[2])
if len(sys.argv)>3:
    fn = sys.argv[3]
else:
    fn = None
data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
view_names = ['axial', 'coronal', 'sagittal']
data_folder_out = '/global/scratch/fanwei_kong/ImageData/%s/2d_multiclass-%s2%s' % (base_name, view_names[view],fn)
overwrite = True 





def data_preprocess(modality,data_folder,view, data_folder_out, comm, rank):
  train_img_path = []
  train_mask_path = []
  train_weights = []
  for m in modality:
    #imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m, fn='_test_nolabel')
    imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m, fn=fn)
    if rank ==0:
      print("number of training data %d" % len(imgVol_fn))
    assert len(imgVol_fn) == len(mask_fn)
    num_vol_per_core = int(np.ceil(len(imgVol_fn)/comm.Get_size()))
    begin = rank*num_vol_per_core
    end = (rank+1)*num_vol_per_core if (rank+1)*num_vol_per_core <= len(imgVol_fn) else len(imgVol_fn)
    imgVol_fn = imgVol_fn[begin:end]
    mask_fn = mask_fn[begin:end] 
    comm.barrier()   
    print("rank %d, begin %d, end %d" % (rank, begin,end)) 

    for i in range(len(imgVol_fn)):
      img_path = imgVol_fn[i]
      mask_path = mask_fn[i]
      imgVol = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # numpy array
      imgVol = RescaleIntensity(imgVol, m)
      #imgVol = HistogramEqualization(imgVol)
      maskVol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))  # numpy array
      maskVol = swapLabels(maskVol)
      if m =="mr":
        imgVol = np.moveaxis(imgVol,0,-1)
        maskVol = np.moveaxis(maskVol,0,-1)
      print("number of image slices in this view %d" % imgVol.shape[view])
      #remove the blank images with a probability - find the index first
      IDs = np.max(np.max(np.moveaxis(maskVol,view,0),axis=-1),axis=-1)==0
      
      for sid in range(imgVol.shape[view]):
        if IDs[sid] and np.random.rand(1)>0.2:
            continue
        
        out_im_path = os.path.join(data_folder_out, m+'_train', m+'_train'+str(range(begin,end)[i])+'_'+str(sid))
        out_msk_path = os.path.join(data_folder_out, m+'_train_masks',  m+'_train_mask'+str(range(begin,end)[i])+'_'+str(sid))
        slice_im = np.moveaxis(imgVol,view,0)[sid,:,:]
        slice_msk = np.moveaxis(maskVol,view,0)[sid,:,:]
        np_to_tfrecords(slice_im.astype(np.float32),slice_msk.astype(np.int64), out_im_path, verbose=True)
        train_img_path.append(out_im_path)
        train_mask_path.append(out_msk_path)
    comm.barrier()
      
  return train_img_path, train_mask_path


if rank == 0:
  print("Making dir...")
  try:
    os.mkdir(data_folder_out)
  except Exception as e: print(e)
  for m in modality:
    try:
      os.mkdir(os.path.join(data_folder_out, m+'_train'))
    except Exception as e: print(e)
comm.barrier()

if overwrite:
  _, _  = data_preprocess(modality,data_folder,view, data_folder_out,comm,rank)
