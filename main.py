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
    fns = glob.glob(os.path.join(os.path.dirname(__file__),"4dct","*.nii.gz"))
    for fn in fns: 
        print(fn)
        #load label map 
        label = label_io.loadLabelMap(fn)
        label = utils.resample(label)
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

def test5():
    """
    Test function to create RV and LV mesh surfaces for electromechanical simulations
    """
    #fns = glob.glob(os.path.join(os.path.dirname(__file__),"examples","*.nii.gz"))
    fns = [os.path.join(os.path.dirname(__file__), "examples", "ct_train_1015_label.nii.gz")]
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
      
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))

        vtkIm = utils.vtkImageResample(vtkIm, (256,256,256), 'linear')
        
        newIm = utils.labelDilateErode(vtkIm, 600, 0, 8)
        #ori = (-30.472927203693008, 217.50936443034828, -99.92209600534021)
        #nrm = (-0.27544302463217574, 0.8246285707645975, 0.4940838597446954)
        ori = (17.398820412524746, 328.4073098038115, -190.07031423467626)
        nrm = (-0.4405409315781873, 0.7659402071382034, 0.468251307198719)
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

def test6():
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
        

def test7():
    """
    Registration of surface mesh point set using Elastix
    Performs 3D image registration and move points based on the computed transform
    """
    START_PHASE = 7
    TOTAL_PHASE = 10
    fn = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/4dct_model/outputpoints.txt'
    MODEL_NAME = 'phase%d.nii.gz.vtk'
    IMAGE_NAME = 'phase%d.nii'
    image_dir = '/Users/fanweikong/Documents/ImageData/4DCCTA/MACS40282_20150504/wall_motion_image_volumes'
    import registration
    
    ids = list(range(START_PHASE,TOTAL_PHASE+1)) + list(range(1,START_PHASE))
    
    # Only need to register N-1 mesh
    for index in ids[:-1]:
        print("REGISTERING FROM %d TO %d " % (index, index%TOTAL_PHASE+1))
        # load surface mesh
        fn = os.path.join(os.path.dirname(__file__),"4dct_model",MODEL_NAME % START_PHASE)
        model = label_io.loadVTKMesh(fn)
        fn_out = os.path.join(os.path.dirname(__file__), "4dct_model", "verts.pts")
    
        #ASSUMING increment is 1
        moving_im_fn = os.path.join(image_dir, IMAGE_NAME % (index%TOTAL_PHASE+1)) 
        fixed_im_fn =os.path.join(image_dir, IMAGE_NAME % START_PHASE)
    
        new_model = registration.point_image_transform(utils.resample(sitk.ReadImage(fixed_im_fn)),
            utils.resample(sitk.ReadImage(moving_im_fn)),
            model,
            fn_out
        )
        #ASSUMING increment is 1
        fn_poly = os.path.join(os.path.dirname(__file__), "4dct_model", MODEL_NAME % (index%TOTAL_PHASE+1))
        label_io.writeVTKPolyData(new_model, fn_poly)
   
if __name__=="__main__":
    test4()
