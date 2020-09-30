import numpy as np
import vtk

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
    
def vtk_resample_to_size(image, new_size, order=1):
    size = image.GetDimensions()
    spacing = image.GetSpacing()
    reference_spacing = np.array(size)/np.array(new_size)*np.array(spacing)
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    resize = vtk.vtkImageReslice()
    resize.SetInputData(image)
    resize.SetBackgroundLevel(0.)
    if order==1:
        resize.SetInterpolationModeToLinear()
    elif order==0:
        resize.SetInterpolationModeToNearest()
    elif order==3:
        resize.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")
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
    resize.SetOutputExtent(img_info['extent'])
    resize.SetOutputSpacing(img_info['spacing'])
    resize.Update()
    im = resize.GetOutput()
    im.SetOrigin(img_info['origin'])
    return im



