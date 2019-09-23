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



def LVCFDModeling():
    """
    This is a test function to generate geometry for fluid simulation (aorta, lv, part of atrium)
    The left atrium is cut normal to the direction defined by the normal of the mitral plane
    The amount of left atrium kept can be adjusted by a scalar factor, 
    which scales the distance between mv plane centroid and la centroid
    """
    FACTOR = 0.5

    fns = [os.path.join(os.path.dirname(__file__),"4dct","phase7.nii.gz")]
    for fn in fns: 
        print(fn)
        #load label map 
        label = label_io.loadLabelMap(fn)

        label = utils.resample(label)
        import SimpleITK as sitk
        pylabel = sitk.GetArrayFromImage(label)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "4dct_model"))
        except Exception as e: print(e)
        #remove myocardium, RV, RA and PA
        for tissue in [1, 4, 5,7]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
        
        #locate centroid of mitral plane
        mv_pts = utils.locateRegionBoundary(vtkIm, 3, 2)
        ctr_mv = np.mean(mv_pts, axis=0)
        #centroid of left atrium
        ctr_la = utils.getCentroid(vtkIm, 2)
        #center and nrm of the cutting plane
        length = np.linalg.norm(ctr_la-ctr_mv)
        nrm_la_mv = (ctr_la - ctr_mv)/length
        nrm_mv_plane = utils.fitPlaneNormal(mv_pts)

        #check normal direction
        if np.dot(nrm_la_mv, nrm_mv_plane)>0:
            nrm = nrm_mv_plane
        else:
            nrm = -1 * nrm_mv_plane
        ori = ctr_mv + length * FACTOR * nrm
        vtkIm = utils.recolorVTKPixelsByPlaneByRegion(vtkIm, ori, nrm, 2, 0)
        # convert to binary
        vtkIm = utils.convertVTK2binary(vtkIm)
        #run marchine cube algorithm
        import marching_cube as m_c
        vtkIm = utils.vtkImageResample(vtkIm, (2.,2.,2.),'linear')
        model = m_c.vtk_marching_cube_multi(vtkIm, 0)
        #model = utils.smoothVTKPolydata(model, 10)
    
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "4dct", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)
        
def LVCFDModeling2():
    """
    Modified test6 to cut on the PolyData directly to create better defined inlet/outlet geometry
    The left atrium is cut normal to the direction defined by the normal of the mitral plane
    The amount of left atrium kept can be adjusted by a scalar factor, 
    which scales the distance between mv plane centroid and la centroid
    """
    FACTOR = 0.5

    fns = [os.path.join(os.path.dirname(__file__),"4dct","phase7.nii.gz")]
    for fn in fns: 
        print(fn)
        #load label map 
        label = label_io.loadLabelMap(fn)

        label = utils.resample(label)
        import SimpleITK as sitk
        pylabel = sitk.GetArrayFromImage(label)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "4dct_model"))
        except Exception as e: print(e)
        #remove myocardium, RV, RA and PA
        for tissue in [1, 4, 5,7]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
        
        # Build Cutter for LA
        for tissue in [3,6]:
            la_label = utils.removeClass(pylabel, tissue, 0)
        la_Im = label_io.exportSitk2VTK(label_io.exportPy2Sitk(la_label, label))
        #locate centroid of mitral plane
        mv_pts = utils.locateRegionBoundary(vtkIm, 3, 2)
        ctr_mv = np.mean(mv_pts, axis=0)
        #centroid of left atrium
        ctr_la = utils.getCentroid(vtkIm, 2)
        #center and nrm of the cutting plane
        length = np.linalg.norm(ctr_la-ctr_mv)
        nrm_la_mv = (ctr_la - ctr_mv)/length
        nrm_mv_plane = utils.fitPlaneNormal(mv_pts)
        #check normal direction
        if np.dot(nrm_la_mv, nrm_mv_plane)>0:
            nrm =  nrm_mv_plane
        else:
            nrm = -1.*nrm_mv_plane

        ori = ctr_mv + length * FACTOR * nrm
        #dilate by a little bit
        la_Im = utils.labelDilateErode(utils.recolorVTKPixelsByPlane(la_Im, ori, -1.*nrm, 0), 2, 0, 1)
        la_Im = utils.convertVTK2binary(la_Im)
        import marching_cube as m_c
        la_cutter = m_c.vtk_marching_cube_multi(la_Im, 0)
        
        fn_poly = os.path.join(os.path.dirname(__file__), "4dct", "la.vtk")
        label_io.writeVTKPolyData(la_cutter, fn_poly)

        # convert to binary
        vtkIm = utils.convertVTK2binary(vtkIm)
        #run marchine cube algorithm
        vtkIm = utils.vtkImageResample(vtkIm, (2.,2.,2.),'linear')
        model = m_c.vtk_marching_cube_multi(vtkIm, 0)
        model = utils.cutPolyDataWithAnother(model, la_cutter,False)
    
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "4dct", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)

