import os
import numpy as np
import glob
import SimpleITK as sitk
import vtk
import label_io
import utils
from image_processing import lvImage
import models
class Registration:
    """
    Class to perform 3D image registration
    """
    def __init__(self, fixed_im_fn=None, moving_im_fn=None, fixed_mask_fn=None, smooth=False):
        """

        Args:
            fixed_im_fn: target image fn
            moving_im_fn: moving image fn
        """
        self.fixed_fn = fixed_im_fn
        self.moving_fn = fixed_im_fn
        self.fixed = None
        self.moving = None
        self.fixed_mask_fn = fixed_mask_fn
    #    self.moving_mask_fn = moving_mask_fn
        self.fixed_mask = None
    #    self.moving_mask = None
        self.parameter_map = None
        self.smooth = smooth

    def updateMovingImage(self, moving_im_fn):
        self.moving_fn = moving_im_fn
        self.moving = None
        self.parameter_map = None

    def updateFixedImage(self, fixed_im_fn):
        self.fixed_fn = fixed_im_fn
        self.fixed = None
        self.parameter_map = None
    
    #def updateMovingMask(self, moving_mask_fn):
    #    self.moving_mask_fn = moving_mask_fn
    #    self.moving_mask = None
    #    self.parameter_map = None

    def updateFixedMask(self, fixed_mask_fn):
        self.fixed_mask_fn = fixed_mask_fn
        self.fixed_mask = None
        self.parameter_map = None

    def loadImages(self):
        self.fixed = sitk.ReadImage(self.fixed_fn)
        #self.fixed_mask = sitk.Cast(sitk.ReadImage(self.fixed_mask_fn), sitk.sitkUInt8 )
        self.moving = sitk.ReadImage(self.moving_fn)
        #self.moving_mask = sitk.Cast(sitk.ReadImage(self.moving_mask_fn),sitk.sitkUInt8 )
        if self.smooth:
            self.fixed = utils.closing(self.fixed, [7, 6, 5, 4, 3, 2, 1])
            self.moving = utils.closing(self.moving, [ 7, 6, 5, 4, 3, 2, 1])
        #fixed = lvImage(self.fixed_fn)
        #moving = lvImage(self.moving_fn)
        #fixed.process([1, 4, 5, 7])
        #moving.process([1, 4, 5, 7])
        #self.fixed = label_io.exportVTK2Sitk(fixed.label)
        #self.moving = label_io.exportVTK2Sitk(moving.label)
        res = np.array(self.fixed.GetSpacing())
        res = np.min(res)/res * 0.8
        self.fixed = utils.resample(self.fixed, res, order=1)
        #self.fixed_mask = utils.resample(self.fixed_mask)
        #self.fixed_mask.SetOrigin(self.fixed.GetOrigin())
        self.moving = utils.resample(self.moving, res, order=1)
        #self.moving_mask = utils.resample(self.moving_mask)
        #self.moving_mask.SetOrigin(self.moving.GetOrigin())

    def computeTransform(self):

        if (self.fixed is None) or (self.moving is None):
            self.loadImages()
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(self.fixed)
        #elastixImageFilter.SetFixedMask(self.fixed_mask)
        p_map_1 = sitk.GetDefaultParameterMap('translation')
        p_map_2 = sitk.GetDefaultParameterMap('affine')
        p_map_3 = sitk.GetDefaultParameterMap('bspline')
        #p_map_3['MaximumNumberOfIterations'] = ['256']
        p_map_3['FinalGridSpacingInPhysicalUnits'] = []
        #p_map_3["MaximumNumberOfSamplingAttempts"] = ['4']
        p_map_3["FinalGridSpacingInVoxels"] = ['12']
        #p_map_3['FinalBSplineInterpolationOrder'] = ['3']
        sitk.PrintParameterMap(p_map_3)
        elastixImageFilter.SetParameterMap(p_map_1)
        elastixImageFilter.AddParameterMap(p_map_2)
        elastixImageFilter.AddParameterMap(p_map_3)
        elastixImageFilter.SetMovingImage(self.moving)
        #elastixImageFilter.SetMovingMask(self.moving_mask)
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
    def polydata_image_transform(self, model, fn, im_out_fn, fn_paras=None):
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
        result_im = transformixImageFilter.GetResultImage()
        sitk.WriteImage(result_im, im_out_fn)
        # build VTK PolyData
        pts = label_io.readElastixPointOuptut(os.path.join(os.path.dirname(fn),'outputpoints.txt'))

        new_poly = vtk.vtkPolyData()
        new_poly.DeepCopy(model.poly)
        new_poly.SetPoints(pts)
        return models.leftVentricle(model.update(new_poly), model.edge_size)


        

