import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from utils import np_to_tfrecords

def HistogramEqualization(pyIm):
    pyImNew = np.empty(pyIm.shape)

    num = int(np.max(pyIm.flatten())-np.min(pyIm.flatten()))+1
    im_hist, bins = np.histogram(pyIm.flatten(), num)

    cdf = im_hist.cumsum()
    cdf = max(pyIm.flatten()) * cdf /cdf[-1]

    image_equalized = np.interp(pyIm.flatten(),bins[:-1],cdf)
    pyImNew = np.reshape(image_equalized,pyIm.shape)
    
    return pyImNew
  
def resample(sitkIm_fn, resolution = (0.5, 0.5, 0.5), dim=3):
  image = sitk.ReadImage(sitkIm_fn)
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(sitk.sitkLinear)
  resample.SetOutputDirection(image.GetDirection())
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputSpacing(resolution)

  orig_size = np.array(image.GetSize(), dtype=np.int)
  orig_spacing = np.array(image.GetSpacing())
  new_size = orig_size*(orig_spacing/np.array(resolution))
  new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
  new_size = [int(s) for s in new_size]
  resample.SetSize(new_size)
  newimage = resample.Execute(image)
  
  return newimage

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
  
def swapLabelsBack(labels,pred):
    labels[labels==421]=420
    unique_label = np.unique(labels)
    new_label = range(len(unique_label))

    for i in range(len(unique_label)):
      pred[pred==i] = unique_label[i]
      
    return pred
    

def RescaleIntensity(slice_im,m,limit):
  #slice_im: numpy array
  #m: modality, ct or mr
  if m =="ct":
    slice_im[slice_im>limit[0]] = limit[0]
    slice_im[slice_im<limit[1]] = limit[1]
    slice_im = slice_im/limit[0]
  elif m=="mr":
    slice_im[slice_im>1500] = 1500
    slice_im = (slice_im-750)/750
  return slice_im
    
def data_preprocess_test(image_vol_fn, view, size, m):
    image_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_vol_fn))
    original_shape = image_vol.shape
    image_vol = RescaleIntensity(image_vol, m)
    shape = [size, size, size]
    shape[view] = image_vol.shape[view]
    image_vol_resize = resize(image_vol, tuple(shape))
    
    return image_vol_resize, original_shape
  
def data_preprocess(modality,data_folder,view, data_folder_out):
  train_img_path = []
  train_mask_path = []
  train_weights = []
  for m in modality:
    imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m)
    print("number of training data %d" % len(imgVol_fn))
    assert len(imgVol_fn) == len(mask_fn)

    for i in range(0,len(imgVol_fn)):
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
      IDs = np.max(maskVol,axis=view)==0
      
      for sid in range(imgVol.shape[view]):
        if IDs[sid] and np.random.rand(1)>0.2:
            continue
        out_im_path = os.path.join(data_folder_out, m+'_train', m+'_train'+str(i)+'_'+str(sid))
        out_msk_path = os.path.join(data_folder_out, m+'_train_masks',  m+'_train_mask'+str(i)+'_'+str(sid))
        slice_im = np.moveaxis(imgVol,view,0)[sid,:,:]
        slice_msk = np.moveaxis(maskVol,view,0)[sid,:,:]
        np_to_tfrecords(slice_im.astype(np.float32),slice_msk.astype(np.int64), out_im_path, verbose=True)
        train_img_path.append(out_im_path)
        train_mask_path.append(out_msk_path)
  return train_img_path, train_mask_path
  
