import os
import numpy as np
import SimpleITK as sitk
import vtk
import label_io
import models
class Registration:
    """
    Class to perform 3D image registration
    """
    def __init__(self, fixed_im_fn=None, moving_im_fn=None):
        """

        Args:
            fixed_im_fn: target image fn
            moving_im_fn: moving image fn
        """
        self.fixed_fn = fixed_im_fn
        self.moving_fn = fixed_im_fn
        self.fixed = None
        self.moving = None
        self.registration_filter = None

    def updateMovingImage(self, moving_im_fn):
        self.moving_fn = moving_im_fn
        self.moving = None

    def updateFixedImage(self, fixed_im_fn):
        self.fixed_fn = fixed_im_fn
        self.fixed = None

    def loadImages(self):
        self.fixed = sitk.ReadImage(self.fixed_fn)
        self.moving = sitk.ReadImage(self.moving_fn)

    def computeTransform(self):

        if (self.fixed is None) or (self.moving is None):
            self.loadImages()
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_im)
        elastixImageFilter.SetMovingImage(moving_im)
        elastixImageFilter.Execute()

        self.registration_filter = elastixImageFilter

    def polydata_image_transform(self, model, fn):
        """
        Transform the points of a geometry using the computed transformation
        
        Args:
            poly: surface mesh to transform (vtk PolyData)
            fn: file name to write the vertices of polydata to file

        Returns:
            new_poly: transformed surface mesh (vtk PolyData)
        """

        label_io.writeVTKPolyDataVerts(model.poly, fn)
        if self.registration_filter is None:
            self.computeTransform()
        # wrap point set
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(moving_im)
        transformixImageFilter.SetTransformParameterMap(self.registration_filter.GetTransformParameterMap())
        transformixImageFilter.SetFixedPointSetFileName(fn)
        transformixImageFilter.SetOutputDirectory(os.path.dirname(fn))
        transformixImageFilter.Execute()

        # build VTK PolyData
        pts = label_io.readElastixPointOuptut(os.path.join(os.path.dirname(fn),'outputpoints.txt'))

        new_poly = vtk.vtkPolyData()
        new_poly.DeepCopy(poly)
        new_poly.SetPoints(pts)
        return models.leftVentricle(model.update(new_poly))


        

