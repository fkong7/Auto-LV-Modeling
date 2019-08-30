import os
import numpy as np
import SimpleITK as sitk
import vtk
import label_io

def point_image_transform(fixed_im, moving_im, poly, fn):
    """
    Register one 3D volume to another, 
    Transform the points using the computed transformation

    Args:
        fixed_im: target image (sitk Image)
        moving_im: moving image (sikt Image)
        poly: surface mesh to transform (vtk PolyData)
        fn: file name to write the vertices of polydata to file

    Returns:
        new_poly: transformed surface mesh (vtk PolyData)
    """
    label_io.writeVTKPolyDataVerts(poly, fn)
    # Compute the transformation
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_im)
    elastixImageFilter.SetMovingImage(moving_im)
    elastixImageFilter.Execute()

    # wrap point set

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(moving_im)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetFixedPointSetFileName(fn)
    transformixImageFilter.SetOutputDirectory(os.path.dirname(fn))
    transformixImageFilter.Execute()

    # build VTK PolyData
    pts = label_io.readElastixPointOuptut(os.path.join(os.path.dirname(fn),'outputpoints.txt'))

    new_poly = vtk.vtkPolyData()
    new_poly.DeepCopy(poly)
    new_poly.SetPoints(pts)
    return new_poly

