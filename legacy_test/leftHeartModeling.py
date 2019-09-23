import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "../src"))

import glob
import numpy as np
import label_io
from marching_cube import marching_cube, vtk_marching_cube
from plot import plot_surface
import utils
import vtk


def leftHeartModeling():
    """
    This is a test funciton to create manifold mesh surfaces for blood pool with vtk marching cube
    """
    fns = glob.glob(os.path.join(os.path.dirname(__file__),"4dct","*.nii.gz"))
    for fn in fns: 
        print(fn)
        #load label map 
        label = label_io.loadLabelMap(fn)
        label = utils.resample(label)
        import SimpleITK as sitk
        pylabel = sitk.GetArrayFromImage(label)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "4dct_model_raw"))
        except Exception as e: print(e)
        #remove myocardium, RV, RA and PA
        for tissue in [1, 4, 5,7]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        pylabel = utils.convert2binary(label_io.exportPy2Sitk(pylabel, label))
        pylabel = utils.eraseBoundary(pylabel, 3, 0)
        
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
    
        #run marchine cube algorithm
        import marching_cube as m_c
        model = m_c.vtk_marching_cube_multi(vtkIm, 0)
        model = utils.smoothVTKPolydata(model, 1000)
    
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "4dct_model_raw", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)

