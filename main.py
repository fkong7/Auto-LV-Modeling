import os
import glob
import numpy as np
import label_io
from marching_cube import marching_cube, vtk_marching_cube
from plot import plot_surface
import utils
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import vtk

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

def test3():
    """
    This is a test funciton to create multi-class surface mesh from label maps with the marhing cube function from vtk
    """
    fn = os.path.join(os.path.dirname(__file__), "examples", "ct_train_1002_label.nii.gz")
    
    #load label map 
    label = label_io.loadLabelMap(fn)
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

def test4():
    """
    This is a test funciton to create manifold mesh surfaces for blood pool with vtk marching cube
    """
    fns = glob.glob(os.path.join(os.path.dirname(__file__),"examples","*.nii.gz"))
    for fn in fns: 
        print(fn)
        #load label map 
        label = label_io.loadLabelMap(fn)
        pylabel = sitk.GetArrayFromImage(label)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
        except Exception as e: print(e)
        #remove myocardium, RV, RA and PA
        for tissue in [205, 600, 550, 850]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        pylabel = utils.convert2binary(label_io.exportPy2Sitk(pylabel, label))
        pylabel = utils.eraseBoundary(pylabel, 3, 0)
        
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
    
        #run marchine cube algorithm
        import marching_cube as m_c
        model = m_c.vtk_marching_cube_multi(vtkIm, 0)
    
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)

def test5():
    """
    Test function to create RV and LV mesh surfaces for electromechanical simulations
    """
    #fns = glob.glob(os.path.join(os.path.dirname(__file__),"examples","*.nii.gz"))
    fns = [os.path.join(os.path.dirname(__file__), "examples", "ct_train_1019_label.nii.gz")]
    for fn in fns: 
        print(fn)
    
        #load label map 
        label = label_io.loadLabelMap(fn)
        pylabel = sitk.GetArrayFromImage(label)
        #remove myocardium, RV, RA and PA
        for tissue in [500, 420, 550, 820,850]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        #debug: write to disk
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
        except Exception as e: print(e)
      
        fn_out2 = os.path.join(os.path.dirname(__file__), "debug", "test_volume_multi2.vti")
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
        # debug
        fn_out3 = os.path.join(os.path.dirname(__file__), "debug", "test_volume_multi3.vti")
        vtkIm = utils.vtkImageResample(vtkIm, (10,10,10), 'linear')
        label_io.writeVTKImage(vtkIm, fn_out3)

        ori = (-30.472927203693008, 217.50936443034828, -99.92209600534021)
        nrm = (-0.27544302463217574, 0.8246285707645975, 0.4940838597446954)
        vtkIm = utils.recolorVTKPixelsByPlane(vtkIm, ori, nrm, 0)
        label_io.writeVTKImage(vtkIm, fn_out2)
        
        newIm = utils.labelDilateErode(vtkIm, 600, 0, 5)
        
        #run marchine cube algorithm
        import marching_cube as m_c
        model = m_c.vtk_marching_cube_multi(newIm, 0)
        model = utils.clipVTKPolyData(model, ori, nrm)

        
        #write to vtk polydata
        fn_poly = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(fn)+".vtk")
        label_io.writeVTKPolyData(model, fn_poly)
if __name__=="__main__":
    test5()
