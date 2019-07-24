import os
import numpy as np
import label_io
from marching_cube import marching_cube
from plot import plot_surface
import utils
from scipy.ndimage import gaussian_filter

def test1():
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
    #pylabel = gaussian_filter(pylabel, sigma=1)
    output = marching_cube(pylabel, 0.99 )
    verts, faces, _, _ = output
    #write to vtk polydata
    fn_poly = os.path.join(os.path.dirname(__file__), "debug", "test_poly.vtk")
    label_io.writeVTKPolyData(label_io.isoSurf2VTK(verts, faces), fn_poly)
    #plot
    plot_surface(verts, faces, smoothed.shape)


def test2():
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
    fn_out = os.path.join(os.path.dirname(__file__), "debug", "test_volume.vti")
    label_io.writeVTKImage(label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label)), fn_out)
    
    #run marchine cube algorithm
    #pylabel = gaussian_filter(pylabel, sigma=1)
    output = marching_cube(pylabel, 0.99 )
    verts, faces, _, _ = output
    #write to vtk polydata
    fn_poly = os.path.join(os.path.dirname(__file__), "debug", "test_poly.vtk")
    label_io.writeVTKPolyData(label_io.isoSurf2VTK(verts, faces), fn_poly)
    #plot
    plot_surface(verts, faces, smoothed.shape)


if __name__=="__main__":
    test2()
