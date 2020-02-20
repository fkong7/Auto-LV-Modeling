import os
import numpy as np
import glob
import SimpleITK as sitk
import vtk
import label_io
from image_processing import lvImage
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
        self.parameter_map = None

    def updateMovingImage(self, moving_im_fn):
        self.moving_fn = moving_im_fn
        self.moving = None
        self.parameter_map = None

    def updateFixedImage(self, fixed_im_fn):
        self.fixed_fn = fixed_im_fn
        self.fixed = None
        self.parameter_map = None

    def loadImages(self):
#        fixed = lvImage(self.fixed_fn)
#        moving = lvImage(self.moving_fn)
#        fixed.process([1, 4, 5, 7])
#        moving.process([1, 4, 5, 7])
#        self.fixed = label_io.exportVTK2Sitk(fixed.label)
#        self.moving = label_io.exportVTK2Sitk(moving.label)
        self.fixed = sitk.ReadImage(self.fixed_fn)
        self.moving = sitk.ReadImage(self.moving_fn)
        self.fixed = utils.closing(self.fixed, [7, 6, 5, 4, 3, 2, 1])
        self.moving = utils.closing(self.moving, [7, 6, 5, 4, 3, 2, 1])


    def computeTransform(self):

        if (self.fixed is None) or (self.moving is None):
            self.loadImages()
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(self.fixed)
        elastixImageFilter.SetMovingImage(self.moving)
        elastixImageFilter.Execute()

        self.parameter_map = elastixImageFilter.GetTransformParameterMap()

    def writeParameterMap(self, fn):
        if self.parameter_map is None:
            return
        for i, para_map in enumerate(self.parameter_map):
            para_map_fn = os.path.splitext(fn)[0]+'_%d.txt' % i
            sitk.WriteParameterFile(para_map, para_map_fn)

    def readParameterMap(self, fn):
        fns = sorted(glob.glob(os.path.splitext(fn)[0]+"*"))
        if len(fns)==0:
            raise IOError("No Transformation file found")
        map_list = list()
        for para_map_fn in fns:
            map_list.append(sitk.ReadParameterFile(para_map_fn))
        self.parameter_map=tuple(map_list)
    def polydata_image_transform(self, model, fn, fn_paras=None):
        """
        Transform the points of a geometry using the computed transformation
        
        Args:
            poly: surface mesh to transform (vtk PolyData)
            fn: file name to write the vertices of polydata to file
            fn_paras: file name to the parameter map of previously done registration
        Returns:
            new_poly: transformed surface mesh (vtk PolyData)
        """

        label_io.writeVTKPolyDataVerts(model.poly, fn)
        if self.parameter_map is None:
            try:
                self.readParameterMap(fn_paras)
            except Exception as e:
                self.computeTransform()
        if (self.fixed is None) or (self.moving is None):
            self.loadImages()

        # wrap point set
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(self.moving)
        transformixImageFilter.SetTransformParameterMap(self.parameter_map)
        transformixImageFilter.SetFixedPointSetFileName(fn)
        transformixImageFilter.SetOutputDirectory(os.path.dirname(fn))
        transformixImageFilter.Execute()

        # build VTK PolyData
        pts = label_io.readElastixPointOuptut(os.path.join(os.path.dirname(fn),'outputpoints.txt'))

        new_poly = vtk.vtkPolyData()
        new_poly.DeepCopy(model.poly)
        new_poly.SetPoints(pts)
        return models.leftVentricle(model.update(new_poly))


        

