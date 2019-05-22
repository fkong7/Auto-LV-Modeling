import os
import glob
import sys

import numpy as np
import SimpleITK as sitk
print('Importing tf...')
import tensorflow as tf
from mpi4py import MPI
from utils import np_to_tfrecords
from utils import getTrainNLabelNames
from preProcess import swapLabels, RescaleIntensity, HistogramEqualization
import argparse

print('Start...')
parser = argparse.ArgumentParser()
parser.add_argument('--folder', nargs=1, help='Name of the folder containing the image data')
parser.add_argument('--view', nargs=1, type=int, help='Which views 0, 1, 2, axial, coronal, sagittal')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--out_folder', nargs='?', default='_train', help='Folder postfix of the folder to look for')
parser.add_argument('--n_channel', nargs='?', const=1, default=1, type=int, help='Number of channels')
parser.add_argument('--intensity',nargs='+', type=int, default=[750,-750], help='Intensity range to clip to [upper, lower]')
print('Finished parsing...')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total = comm.Get_size()
"""# Set up"""
args = parser.parse_args()

modality = args.modality
base_name = args.folder[0]
view = args.view[0]
fn = args.out_folder
channel = args.n_channel
intensity = args.intensity

data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
view_names = ['axial', 'coronal', 'sagittal']
data_folder_out = '/global/scratch/fanwei_kong/ImageData/%s/2d_multiclass-%s2%s' % (base_name, view_names[view],fn)

if channel>1:
    data_folder_out += '_multi%d' %  channel
    




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
    

    num_vol_per_core = int(np.floor(len(imgVol_fn)/comm.Get_size()))
    extra = len(imgVol_fn) % comm.Get_size()
    vol_ids = list(range(rank*num_vol_per_core,(rank+1)*num_vol_per_core))
    if rank < extra:
        vol_ids.append(len(imgVol_fn)-1-rank)

    imgVol_fn = [imgVol_fn[k] for k in vol_ids]
    mask_fn = [mask_fn[k] for k in vol_ids]
    print(vol_ids)

    for i in range(len(imgVol_fn)):
      img_path = imgVol_fn[i]
      mask_path = mask_fn[i]
      imgVol = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # numpy array
      imgVol = HistogramEqualization(imgVol)
      imgVol = RescaleIntensity(imgVol, m, intensity)
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
        out_im_path = os.path.join(data_folder_out, m+'_train', m+'_train'+str(vol_ids[i])+'_'+str(sid))
        out_msk_path = os.path.join(data_folder_out, m+'_train_masks',  m+'_train_mask'+str(vol_ids[i])+'_'+str(sid))
        up = int((channel-1)/2)
        down = int(channel-up)
        if sid+down >= imgVol.shape[view] or sid-up < 0:
            print(sid, down, up)
            continue
        slice_im = np.moveaxis(imgVol,view,0)[sid-up:sid+down,:,:]
        slice_msk = np.moveaxis(maskVol,view,0)[sid,:,:]
        if slice_im.shape[0]!=channel:
            print("Image channel size is incorrect!")
            continue
        np_to_tfrecords(slice_im.astype(np.float32),slice_msk.astype(np.int64), out_im_path, verbose=True)
        train_img_path.append(out_im_path)
        train_mask_path.append(out_msk_path)
      
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

data_preprocess(modality,data_folder,view, data_folder_out,comm,rank)
