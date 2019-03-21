import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from utils import np_to_tfrecords



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
    
def data_preprocess_test(image_vol_fn, view, size, m):
    image_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_vol_fn))
    original_shape = image_vol.shape
    image_vol = RescaleIntensity(image_vol, m)
    shape = [size, size, size]
    shape[view] = image_vol.shape[view]
    image_vol_resize = resize(image_vol, tuple(shape))
    
    return image_vol_resize, original_shape
  
  
def data_preprocess(modality,data_folder,view, data_folder_out):
  for m in modality:
    imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m)
    print("number of training data %d" % len(imgVol_fn))
    assert len(imgVol_fn) == len(mask_fn)

    train_img_path = []
    train_mask_path = []
    for i in range(0,len(imgVol_fn)):
      img_path = imgVol_fn[i]
      mask_path = mask_fn[i]
      imgVol = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # numpy array
      #imgVol = HistogramEqualization(imgVol)
      maskVol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))  # numpy array
      maskVol = swapLabels(maskVol)
      print("number of image slices in this view %d" % imgVol.shape[view])
      for sid in range(imgVol.shape[view]):
        out_im_path = os.path.join(data_folder_out, m+'_train', m+'_train'+str(i)+'_'+str(sid))
        out_msk_path = os.path.join(data_folder_out, m+'_train_masks',  m+'_train_mask'+str(i)+'_'+str(sid))
        slice_im = np.moveaxis(imgVol,view,0)[sid,:,:]
        slice_msk = np.moveaxis(maskVol,view,0)[sid,:,:]
        #slice_im = HistogramEqualization(slice_im)
        #sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(sitk.GetImageFromArray(RescaleIntensity(slice_im,m).astype(np.uint16))), sitk.sitkUInt8),out_im_path+'.png')
        #sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(sitk.GetImageFromArray((slice_msk).astype(np.uint16))), sitk.sitkUInt8), out_msk_path+'.png')
        #np.save(out_im_path+'.npy',RescaleIntensity(slice_im,m))
        #np.save(out_msk_path+'.npy',slice_msk)
        np_to_tfrecords(RescaleIntensity(slice_im,m).astype(np.float32),slice_msk.astype(np.int64), out_im_path, verbose=True)
        #tfio.np_to_tfrecords(slice_msk.astype(np.float32), None, out_msk_path, verbose=True)
        train_img_path.append(out_im_path)
        train_mask_path.append(out_msk_path)
  return train_img_path, train_mask_path