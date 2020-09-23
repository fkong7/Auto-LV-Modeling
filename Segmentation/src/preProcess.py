import numpy as np
import SimpleITK as sitk
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
    
    from skimage.transform import resize
    image_vol = resize(image_vol, (space[0]*original_shape[0],space[1]*original_shape[1], space[2]*original_shape[2])) 
    

    
    image_vol = RescaleIntensity(image_vol, m, [750,-750])
    
    image_info = (ori, space, direc)
    
    return image_vol, original_shape, image_info


def HistogramEqualization(pyIm):
    pyImNew = np.empty(pyIm.shape)

    num = int(np.max(pyIm.flatten())-np.min(pyIm.flatten()))+1
    im_hist, bins = np.histogram(pyIm.flatten(), num)

    cdf = im_hist.cumsum()
    cdf = max(pyIm.flatten()) * cdf /cdf[-1]

    image_equalized = np.interp(pyIm.flatten(),bins[:-1],cdf)
    pyImNew = np.reshape(image_equalized,pyIm.shape)
    
    return pyImNew
  
def Resize_by_view(image_vol, view, size):
    shape = [size, size, size]
    shape[view] = image_vol.shape[view]
    from skimage.transform import resize
    image_vol_resize = resize(image_vol.astype(float), tuple(shape))
    return image_vol_resize

  
def resample(sitkIm, resolution = (0.5, 0.5, 0.5),order=1,dim=3):
  if type(sitkIm) is str:
    image = sitk.ReadImage(sitkIm)
  else:
    image = sitkIm
  resample = sitk.ResampleImageFilter()
  if order==1:
    resample.SetInterpolator(sitk.sitkLinear)
  else:
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
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
    if len(blank)==0:
        continue
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

def reference_image_build(spacing, size, template_size, dim):
    #template size: image(array) dimension to resize to: a list of three elements
  reference_size = template_size
  reference_spacing = np.array(size)/np.array(template_size)*np.array(spacing)
  reference_spacing = np.mean(reference_spacing)*np.ones(3)
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

def resample_spacing(sitkIm, resolution=(0.5, 0.5, 0.5), dim=3, template_size=(256, 256, 256), order=1):
  print("ok")
  if type(sitkIm) is str:
    image = sitk.ReadImage(sitkIm)
  else:
    image = sitkIm
  orig_direction = image.GetDirection()
  orig_size = np.array(image.GetSize(), dtype=np.int)
  orig_spacing = np.array(image.GetSpacing())
  new_size = orig_size*(orig_spacing/np.array(resolution))
  new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
  new_size = [int(s) for s in new_size]
  new_size = np.abs(np.matmul(np.reshape(orig_direction, (3,3)), np.array(new_size)))
  ref_img = reference_image_build(resolution, new_size, template_size, dim)
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

def scale(sitkIm, transform, scale_factor = [1., 1., 1.]):
    dim = sitkIm.GetDimension()
    new_transform = sitk.AffineTransform(transform)
    scale = np.eye(dim)
    np.fill_diagonal(scale, 1./np.array(scale_factor))
    matrix = np.array(transform.GetMatrix()).reshape((dim,dim))
    matrix = np.matmul(matrix, scale)
    new_transform.SetMatrix(matrix.ravel())
    new_transform.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
    return new_transform

def translate(sitkIm, transform, translate=[0.,0.,0.]):
    affine = sitk.AffineTransform(transform)
    affine.SetTranslation(translate)
    affine.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
    return affine

def rotation(sitkIm, transform, angles = [0.,0.,0.]):
    rads = np.array(angles)/180.*np.pi
    dim = sitkIm.GetDimension()
    x_rot = np.eye(dim)
    x_rot = [[1., 0., 0.], [0., np.cos(rads[0]), -np.sin(rads[0])], [0., np.sin(rads[0]), np.cos(rads[0])]]
    y_rot = [[np.cos(rads[1]), 0., np.sin(rads[1])], [0.,1.,0.], [-np.sin(rads[1]), 0., np.cos(rads[1])]]
    z_rot = [[np.cos(rads[2]), -np.sin(rads[2]), 0.], [np.sin(rads[2]), np.cos(rads[2]), 0.], [0., 0., 1.]]
    rot_matrix = np.matmul(np.matmul(np.array(x_rot), np.array(y_rot)), np.array(z_rot))
    matrix = np.array(transform.GetMatrix()).reshape((dim, dim))
    matrix = np.matmul(matrix, rot_matrix)
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetMatrix(matrix.ravel())
    new_transform.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
    return new_transform

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
    if type(slice_im) != np.ndarray:
        raise RuntimeError("Input image is not numpy array")
    #slice_im: numpy array
    #m: modality, ct or mr

    mean = np.mean(slice_im)
    std = np.std(slice_im)
    slice_im = (slice_im-mean)/std/2.

    print("Intensity range: ", np.max(slice_im), np.min(slice_im))

    return slice_im
    


