import numpy as np
from utils import np_to_tfrecords
import vtk
  
def resample(sitkIm, resolution = (0.5, 0.5, 0.5),order=1,dim=3):
  import SimpleITK as sitk
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
    
  print("Final shape after cropping: ", mask.shape)
  ratio = np.array(mask.shape)/np.array(ori_shape)
  return mask, ratio


def transform_func(image, reference_image, transform, order=1):
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
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
    import SimpleITK as sitk
    affine = sitk.AffineTransform(transform)
    affine.SetTranslation(translate)
    affine.SetCenter(sitkIm.TransformContinuousIndexToPhysicalPoint(np.array(sitkIm.GetSize())/2.0))
    return affine

def rotation(sitkIm, transform, angles = [0.,0.,0.]):
    import SimpleITK as sitk
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
        newl = new_label[i]
        labels[labels==label] = newl
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
    if m =="ct":
        rng = abs(limit[0]-limit[1])
        threshold = rng/2
        slice_im[slice_im>limit[0]] = limit[0]
        slice_im[slice_im<limit[1]] = limit[1]
        slice_im = slice_im/threshold
    elif m=="mr":
        pls = np.unique(slice_im)
        upper = np.percentile(slice_im, 99)
        lower = np.percentile(slice_im, 20)
        slice_im[slice_im>upper] = upper
        slice_im[slice_im<lower] = lower
        slice_im -= int(lower)
        rng = upper - lower
        slice_im = slice_im/rng*2
        slice_im -= 1
    return slice_im
    
def vtk_reslice_image(image, ori, order=1):
    reslice = vtk.vtkImageReslice()
    ori.Invert()
    if order==1:
        reslice.SetInterpolationModeToLinear()
    else:
        reslice.SetInterpolationModeToNearestNeighbor()
    reslice.SetInputData(image)
    reslice.SetResliceAxes(ori)
    reslice.BorderOn()
    reslice.Update()
    return reslice.GetOutput()

def vtk_resample_to_size(image, new_size, order=1):
    interp = vtk.vtkImageInterpolator()
    if order==1:
        interp.SetInterpolationModeToLinear()
    elif order==0:
        interp.SetInterpolationModeToNearest()
    elif order==3:
        interp.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")
    size = image.GetDimensions()
    spacing = image.GetSpacing()
    reference_spacing = np.array(size)/np.array(new_size)*np.array(spacing)
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    resize = vtk.vtkImageReslice()
    resize.SetInputData(image)
    resize.SetBackgroundLevel(0.)
    resize.SetInterpolator(interp)
    resize.SetOutputSpacing(reference_spacing)
    resize.SetOutputExtent(0, new_size[0]-1, 0, new_size[1]-1, 0, new_size[2]-1)
    resize.Update()
    return resize.GetOutput()

def vtk_resample_with_info_dict(image, img_info, order=1):
    interp = vtk.vtkImageInterpolator()
    if order==1:
        interp.SetInterpolationModeToLinear()
    elif order==0:
        interp.SetInterpolationModeToNearest()
    elif order==3:
        interp.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")
    size = image.GetDimensions()
    reference_spacing = np.array(img_info['size'])/np.array(size)*np.array(img_info['spacing'])
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    image.SetSpacing(reference_spacing)
    resize = vtk.vtkImageReslice()
    resize.SetInputData(image)
    resize.SetInterpolator(interp)
    resize.SetBackgroundLevel(0.)
    #resize.SetOutputOrigin(img_info['origin'])
    resize.SetOutputExtent(img_info['extent'])
    resize.SetOutputSpacing(img_info['spacing'])
    resize.Update()
    im = resize.GetOutput()
    im.SetOrigin(img_info['origin'])
    return im


def vtk_resample_image(image, spacing, order=1):
    """
    Resamples the vtk image to the given dimenstion
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    reslicer = vtk.vtkImageResample()
    reslicer.SetInputData(image)
    if order==1:
        reslicer.SetInterpolationModeToLinear()
    elif order==0:
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif order==3:
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()

