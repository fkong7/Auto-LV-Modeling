import os
import numpy as np
import glob
import sys
sys.path.append('../src')
import SimpleITK as sitk
import preProcess
import utils
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
        self.fixed = preProcess.resample_spacing(sitk.ReadImage(self.fixed_fn))[0]
        self.moving = preProcess.resample_spacing(sitk.ReadImage(self.moving_fn))[0]
        #self.fixed = sitk.ReadImage(self.fixed_fn)
        #self.moving = sitk.ReadImage(self.moving_fn)

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
    def image_transform(self, fns, fn_paras=None):
        """
        Transform the points of a geometry using the computed transformation
        
        Args:
            fns: dictionary of the file names: in_im, in_label, out_im, out_label
            fn_paras: file name to the parameter map of previously done registration
        Returns:
            new_label: transformed label 
        """

        self.readParameterMap(fn_paras)

        move_im = preProcess.resample_spacing(sitk.ReadImage(fns['in_im']))[0]
        
        # wrap point set
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(move_im)
        transformixImageFilter.SetTransformParameterMap(self.parameter_map)
        transformixImageFilter.Execute()
        sitk.WriteImage(transformixImageFilter.GetResultImage(), fns['out_im'])
        
        label_im = preProcess.resample_spacing(sitk.ReadImage(fns['in_label']))[0]
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(label_im)
        transformixImageFilter.SetTransformParameterMap(self.parameter_map)
        transformixImageFilter.Execute()
        sitk.WriteImage(transformixImageFilter.GetResultImage(), fns['out_label'])


if __name__ == '__main__':

    fn_dir = '/Users/fanweikong/Documents/ImageData/MMWHS'
    modality = ["ct", "mr"]

    x_fns = []
    y_fns = []
    for m in modality:
        im_fns, label_fns = utils.getTrainNLabelNames(fn_dir, m, ext='*.nii.gz',fn='_train')
        x_fns += im_fns
        y_fns += label_fns

    from itertools import combinations
    register = Registration()
    for comb in combinations(x_fns,2):
        print(comb[0], comb[1])
        register.updateMovingImage(comb[0])
        register.updateFixedImage(comb[1])
        paras_fn = os.path.splitext(comb[0])[0]+'TO'+os.path.splitext(os.path.basename(comb[1]))[0] + '.txt'
        try: 
            register.readParameterMap(paras_fn)
        except:
            try:
                register.computeTransform()
                register.writeParameterMap(paras_fn)
            except Exception as e: print(e)
        print(paras_fn)




