import os
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
import vtk
import utils
import label_io
import SimpleITK as sitk
from image_processing import lvImage

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

def convert_polydata_to_image(poly_fn, image_fn, out_im_fn=None):
    im = utils.vtkImageResample(label_io.loadLabelMap(image_fn), (0.25, 0.25, 0.25), opt='NN')

    #im = label_io.loadLabelMap(image_fn)
    poly = label_io.loadVTKMesh(poly_fn)
    im_out = utils.convertPolyDataToImageData(poly, im)
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    im = utils.vtkImageResample(label_io.loadLabelMap(image_fn), (0.25, 0.25, 0.25), opt='NN')
    #im = label_io.loadLabelMap(image_fn)
    py_im_out = vtk_to_numpy(im_out.GetPointData().GetScalars())
    py_im = vtk_to_numpy(im.GetPointData().GetScalars())
    py_im_out[py_im_out!=0] = py_im[py_im_out!=0]
    im_out.GetPointData().SetScalars(numpy_to_vtk(py_im_out))
    im_out = label_io.exportVTK2Sitk(im_out)
    im_out = centering(im_out, sitk.ReadImage(image_fn), order=0)
    if out_im_fn is not None:
        sitk.WriteImage(im_out, out_im_fn)
def post_process(image_fn, out_im_fn = None):
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    im = sitk.ReadImage(image_fn)
    #spacing = im.GetSpacing()
    ##ids = np.flip(np.unique(sitk.GetArrayFromImage(im)))
    #ids = [500, 420, 820, 205, 550, 600, 850]
    #kernel = [int(round(5./spacing[i])) for i in range(3)]
    ##Max kernel size is 7
    #kernel = [7 if kernel[i]>7 else kernel[i] for i in range(3)]
    #ftr = sitk.BinaryMorphologicalClosingImageFilter()
    #ftr.SetKernelRadius(kernel)
    ##ftr.SafeBorderOn()
    #ftr.SetNumberOfThreads(8)
    #ftr2 = sitk.BinaryMorphologicalOpeningImageFilter()
    #ftr2.SetKernelRadius([2, 2, 2])
    #ftr2.SetNumberOfThreads(8)
    #for i in ids:
    #    if i ==0:
    #        continue
    #    ftr.SetForegroundValue(int(i))
    #    ftr2.SetForegroundValue(int(i))
    #    im = ftr.Execute(ftr2.Execute(im))
    #sitk.WriteImage(im,out_im_fn)
    #image = lvImage(out_im_fn)
    image = lvImage(image_fn)
    IDs = [0,205,420,500,550,600,820,850]
    ids = [1,4,5,7]
    image.process(ids)
    im = image.label
    #convert dilated ao and la back to lv
    im2 = lvImage(image_fn)
    im2.resample((1.2, 1.2, 1.2), 'NN')
    im2 = im2.label
    py_im2 = vtk_to_numpy(im2.GetPointData().GetScalars())
    py_im = vtk_to_numpy(im.GetPointData().GetScalars())
    mask_ao = (py_im==6) & (py_im2==500)
    mask_la = (py_im==2) & (py_im2==500)
    py_im[mask_ao] = 3
    py_im[mask_la] = 3
    for i in ids:
        if i==0:
            continue
        mask = (py_im==0) & (py_im2==IDs[i])
        py_im[mask] = i
    im.GetPointData().SetScalars(numpy_to_vtk(py_im))
    im_out = label_io.exportVTK2Sitk(im)
    im_out = centering(im_out, sitk.ReadImage(image_fn), order=0)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(image_fn))
    elastixImageFilter.SetMovingImage(im_out)
    p_map_1 = sitk.GetDefaultParameterMap('translation')
    p_map_1["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    elastixImageFilter.SetParameterMap(p_map_1)
    elastixImageFilter.Execute()
    im_out = elastixImageFilter.GetResultImage()
    if out_im_fn is not None:
        sitk.WriteImage(im_out, out_im_fn)
if __name__=='__main__':

    im_dir = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble_4_20_seg'
    out_dir = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble_4_20_seg_postprocess5'
    try:
        os.makedirs(out_dir)
    except Exception as e: print(e)
    im_fns = sorted(glob.glob(os.path.join(im_dir, '*.nii.gz')))
    for im_fn in im_fns:
        out_fn = os.path.join(out_dir, os.path.basename(im_fn))
        post_process(im_fn, out_fn)


'''
    im_fn_dir = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble-2-10-2_seg'
    poly_fn_dir = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble-2-10-2_surf/surfaces'
    out_fn_dir = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/results/test_ensemble-2-10-2_surf2seg'
    im_fns = sorted(glob.glob(os.path.join(im_fn_dir, '*.nii.gz')))
    poly_fns = sorted(glob.glob(os.path.join(poly_fn_dir, '*.vtk')))
    try:
        os.makedirs(out_fn_dir)
    except Exception as e: print(e)

    for im_fn, poly_fn in zip(im_fns, poly_fns):
        print(im_fn, poly_fn)
        out_fn = os.path.join(out_fn_dir, os.path.basename(im_fn))
        convert_polydata_to_image(poly_fn, im_fn, out_fn)
'''

