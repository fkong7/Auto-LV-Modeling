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



def biventricleModeling():
    """
    Test function to create RV and LV mesh surfaces for electromechanical simulations
    """
    #fns = glob.glob(os.path.join(os.path.dirname(__file__),"examples","*.nii.gz"))
    fns = [os.path.join(os.path.dirname(__file__), "examples", "ct_train_1015_label.nii.gz")]
    for fn in fns: 
        print(fn)
    
        #load label map 
        label = label_io.loadLabelMap(fn)
        import SimpleITK as sitk
        pylabel = sitk.GetArrayFromImage(label)
        #remove myocardium, RV, RA and PA
        for tissue in [500, 420, 550, 820,850]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
        except Exception as e: print(e)
      
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))

        vtkIm = utils.vtkImageResample(vtkIm, (256,256,256), 'linear')
        
        newIm = utils.createTissueThickness(vtkIm, 600, 0, 8)
        #ori = (-30.472927203693008, 217.50936443034828, -99.92209600534021)
        #nrm = (-0.27544302463217574, 0.8246285707645975, 0.4940838597446954)
        ori = (17.398820412524746, 328.4073098038115, -190.07031423467626)
        nrm = (0.4405409315781873, -0.7659402071382034, -0.468251307198719)
        newIm = utils.recolorVTKPixelsByPlane(newIm, ori, nrm, 0)
        fn_out2 = os.path.join(os.path.dirname(__file__), "debug", "test_volume_multi2.vti")
        label_io.writeVTKImage(newIm, fn_out2)
        
        #run marchine cube algorithm
        import marching_cube as m_c
        model = m_c.vtk_marching_cube_multi(newIm, 0)
        #model = utils.clipVTKPolyData(model, ori, nrm)

        
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)

