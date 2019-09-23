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

def segmentation2Surface():
    """
    This is a test funciton to create surface mesh from label maps with the marhing cube function from skimage
    """
    fn = os.path.join(os.path.dirname(__file__), "examples", "ct_train_1002_label.nii.gz")
    
    #load label map 
    label = label_io.loadLabelMap(fn)
    #convert to binary
    pylabel = utils.convert2binary(label)
    #debug: write to disk
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
    except Exception as e: print(e)
    fn_out = os.path.join(os.path.dirname(__file__), "debug", "test_volume.nii.gz")
    label_io.writeSitkIm(label_io.exportPy2Sitk(pylabel, label), fn_out)
    
    #run marchine cube algorithm
    #from scipy.ndimage import gaussian_filter
    #pylabel = gaussian_filter(pylabel, sigma=1)
    output = marching_cube(pylabel, 0.99 )
    verts, faces, _, _ = output
    #write to vtk polydata
    fn_poly = os.path.join(os.path.dirname(__file__), "debug", "test_poly.vtk")
    label_io.writeVTKPolyData(label_io.isoSurf2VTK(verts, faces), fn_poly)
    #plot
    plot_surface(verts, faces, smoothed.shape)

def segmentation2SurfaceVTK():
    """
    This is a test funciton to create surface mesh from label maps with the marhing cube function from vtk
    """
    fn = os.path.join(os.path.dirname(__file__), "examples", "ct_train_1002_label.nii.gz")
    
    #load label map 
    label = label_io.loadLabelMap(fn)
    #convert to binary
    pylabel = utils.convert2binary(label)
    #debug: write to disk
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
    except Exception as e: print(e)
    fn_out = os.path.join(os.path.dirname(__file__), "debug", "test_volume.vti")
    vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
    label_io.writeVTKImage(vtkIm, fn_out)
    
    #run marchine cube algorithm
    mesh  = vtk_marching_cube(vtkIm, 1)
    #write to vtk polydata
    fn_poly = os.path.join(os.path.dirname(__file__), "debug", "test_poly.vtk")
    label_io.writeVTKPolyData(mesh, fn_poly)

def multiclassSeg2Surface():
    """
    This is a test funciton to create multi-class surface mesh from label maps with the marhing cube function from vtk
    """
    fn = os.path.join(os.path.dirname(__file__), "examples", "ct_train_1002_label.nii.gz")
    
    #load label map 
    label = label_io.loadLabelMap(fn)
    import SimpleITK as sitk
    pylabel = sitk.GetArrayFromImage(label)
    #debug: write to disk
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
    except Exception as e: print(e)
    fn_out = os.path.join(os.path.dirname(__file__), "debug", "test_volume_multi.vti")
    vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
    label_io.writeVTKImage(vtkIm, fn_out)
    
    #run marchine cube algorithm
    model = vtk.vtkPolyData()
    for i in np.unique(pylabel):
        if i==0:
           continue
        mesh  = vtk_marching_cube(vtkIm, i)
        mesh = utils.setCellScalar(utils.fillHole(mesh),i)
        model = utils.appendVTKPolydata(model, mesh)
    

    #write to vtk polydata
    fn_poly = os.path.join(os.path.dirname(__file__), "debug", "test_poly_multi.vtk")
    label_io.writeVTKPolyData(model, fn_poly)

