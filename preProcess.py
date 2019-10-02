import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from utils import np_to_tfrecords

def Map_intensity_from_fn(image_vol_fn, m):
    img = sitk.ReadImage(image_vol_fn)
    image_vol = sitk.GetArrayFromImage(img).astype(float)
    original_shape = image_vol.shape
 
    ori = img.GetOrigin()
    space = img.GetSpacing()
    direc = img.GetDirection()
    
    image_vol = RescaleIntensity(image_vol, m, [750,-750])
    
    image_info = (ori, space, direc)
    
    return image_vol, original_shape, image_info
  
def Resize_and_map_intensity_from_fn(image_vol_fn, m):
    img = sitk.ReadImage(image_vol_fn)
    ori = img.GetOrigin()
    space = img.GetSpacing()
    direc = img.GetDirection()
    image_vol = sitk.GetArrayFromImage(img).astype(float)
    original_shape = image_vol.shape
    
    image_vol = resize(image_vol, (space[0]*original_shape[0],space[1]*original_shape[1], space[2]*original_shape[2])) 
    

    
    image_vol = RescaleIntensity(image_vol, m, [750,-750])
    
    image_info = (ori, space, direc)
    
    return image_vol, original_shape, image_info

def Resize_by_view(image_vol, view, size):
    shape = [size, size, size]
    shape[view] = image_vol.shape[view]
    image_vol_resize = resize(image_vol.astype(float), tuple(shape))
    return image_vol_resize

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

def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
  # transform image volume to orientation of eye(dim)
  dim = ref_img.GetDimension()
  affine = sitk.AffineTransform(dim)
  if target is None:
    target = np.eye(dim)
  
  ori = np.reshape(orig_direction, np.eye(dim).shape)
  target = np.reshape(target, np.eye(dim).shape)
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
  rng = abs(limit[0]-limit[1])
  threshold = rng/2
  if m =="ct":
    slice_im[slice_im>limit[0]] = limit[0]
    slice_im[slice_im<limit[1]] = limit[1]
    #(slice_im-threshold-np.min(slice_im))/threshold
    slice_im = slice_im/threshold
  elif m=="mr":
    slice_im -= np.min(slice_im)
    slice_im[slice_im>rng] = rng
    slice_im = (slice_im-threshold)/threshold
  return slice_im
    
def data_preprocess_test(image_vol_fn, view, size, m):
    image_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_vol_fn))
    original_shape = image_vol.shape
    image_vol = RescaleIntensity(image_vol, m)
    shape = [size, size, size]
    shape[view] = image_vol.shape[view]
    image_vol_resize = resize(image_vol, tuple(shape))
    
    return image_vol_resize, original_shape
  
  
