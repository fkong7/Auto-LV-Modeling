import os
import glob

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total = comm.Get_size()
"""# Set up"""
#TO-DO:allow arbitrary number of cpus, right now only divisible number of cpus are allowed
modality = ["ct","mr"]
base_name = 'MMWHS_small_10'
data_folder = '/global/scratch/fanwei_kong/ImageData/' + base_name
view_names = ['axial', 'coronal', 'sagittal']
view = 0 
data_folder_out = '/global/scratch/fanwei_kong/ImageData/%s/2d_multiclass-%s2' % (base_name, view_names[view])
overwrite = True 



"""Find training data filenames and label filenames"""

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz'):
  x_train_filenames = []
  y_train_filenames = []
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+'_train',ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+'_train_masks',ext))):
      y_train_filenames.append(os.path.realpath(subject_dir))
  return x_train_filenames, y_train_filenames

"""Convert 3D data to 2D data"""

def swapLabels(labels):
    labels[labels==421]=420
    unique_label = np.unique(labels)

    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        label = unique_label[i]
        print(label)
        newl = new_label[i]
        print(newl)
        labels[labels==label] = newl
       
    print(unique_label)

    return labels

def RescaleIntensity(slice_im,m):
  #slice_im: numpy array
  #m: modality, ct or mr
  if m =="ct":
    slice_im[slice_im>750] = 750
    slice_im[slice_im<-750] = -750
    slice_im = slice_im/750
  elif m=="mr":
#     top_10 = np.percentile(slice_im,90)
#     above = slice_im[slice_im>top_10]
#     med = np.median(above)
#     slice_im = slice_im/med
#     slice_im[slice_im>1.] = 1.
#     slice_im = slice_im*2.-1.
    slice_im[slice_im>1500] = 1500
    slice_im = (slice_im-750)/750
  return slice_im

def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            
    if Y is not None:
        assert X.shape == Y.shape
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())
    if Y is not None:
        d_feature['Y'] = _int64_feature(Y.flatten())
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])
            
    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def data_preprocess(modality,data_folder,view, data_folder_out, comm, rank):
  train_img_path = []
  train_mask_path = []
  train_weights = []
  for m in modality:
    imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m)
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
