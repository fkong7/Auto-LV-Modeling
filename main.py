import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "src"))

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

def buildSurfaceModelFromImage(fns, fn_poly):
    """
    Modified test6 to cut on the PolyData directly to create better defined inlet/outlet geometry
    The left atrium is cut normal to the direction defined by the normal of the mitral plane
    The amount of left atrium kept can be adjusted by a scalar factor, 
    which scales the distance between mv plane centroid and la centroid

    Args:
        fns: list containing the paths to images
        fn_poly: output file name
    Returns:
        model: constructed surface mesh (VTK PolyData)
        cap_pts_ids: node ids of the points on the caps
    """
    import SimpleITK as sitk
    FACTOR_LA = 0.7
    FACTOR_AA = 1.2
    MESH_RESOLUTION = (2.,2.,2.)

    for fn in fns: 
        #load label map 
        label = label_io.loadLabelMap(fn)

        label = utils.resample(label)
        pylabel = sitk.GetArrayFromImage(label)
        #remove myocardium, RV, RA and PA
        for tissue in [1, 4, 5,7]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
        
        import marching_cube as m_c
        
        # Build Cutter for LA
        def _buildCutter(label,region_id, adjacent_id, FACTOR, op='valve'):
            """
            Build cutter for aorta and la

            Args:
                label: original SimpleITK image
                op: 'valve' or 'tissue', option for normal direction
            """
            cut_Im = label_io.exportSitk2VTK(label)
            #locate centroid of mitral plane or aortic plane
            pts = utils.locateRegionBoundary(cut_Im, adjacent_id, region_id)
            ctr_valve = np.mean(pts, axis=0)
            #centroid of left atrium or aorta
            ctr = utils.getCentroid(cut_Im, region_id)
            #center and nrm of the cutting plane
            length = np.linalg.norm(ctr-ctr_valve)
            nrm_tissue = (ctr - ctr_valve)/length
            nrm_valve_plane = utils.fitPlaneNormal(pts)
            #check normal direction
            if op=='valve':
                nrm = nrm_valve_plane
            elif op=='tissue':
                nrm = nrm_tissue
            else:
                raise ValueError("Incorrect option")
            if np.dot(nrm_tissue, nrm_valve_plane)<0:
                nrm =  -1 *nrm
            ori = ctr_valve + length * FACTOR * nrm/np.linalg.norm(nrm)
        
            #dilate by a little bit
            cut_Im = utils.labelDilateErode(utils.recolorVTKPixelsByPlane(cut_Im, ori, -1.*nrm, 0), region_id, 0, 4)
            # marching cube
            cutter = m_c.vtk_marching_cube(cut_Im, region_id,50)

            return cutter
        
        la_cutter = _buildCutter(label, 2, 3, FACTOR_LA, op='valve')
        aa_cutter = _buildCutter(label, 6, 3, FACTOR_AA, op='tissue')
        
        # convert to binary
        vtkIm = utils.convertVTK2binary(vtkIm)
        #run marchine cube algorithm
        vtkIm = utils.vtkImageResample(vtkIm, MESH_RESOLUTION,'linear')
        model = m_c.vtk_marching_cube_multi(vtkIm, 0, 50)
        model = utils.cutPolyDataWithAnother(model, la_cutter,False)
        model = utils.cutPolyDataWithAnother(model, aa_cutter,False)
        #improve valve opening geometry
        id_lists,boundaries = utils.getPointIdsOnBoundaries(model)
        for idx, (ids, boundary) in enumerate(zip(id_lists, boundaries)):
            boundary = utils.smoothVTKPolyline(boundary, 2)
            model = utils.projectOpeningToFitPlane(model, ids, boundary.GetPoints(), 3)
            # Remove the free cells and update the point lists
            model, id_lists[idx] = utils.removeFreeCells(model, [idx for sub_l in id_lists for idx in sub_l])
        model = utils.smoothVTKPolydata(utils.cleanPolyData(model, 0.))
        model,cap_pts_ids = utils.capPolyDataOpenings(model, 1.5)
        #write to vtk polydata
        label_io.writeVTKPolyData(model, fn_poly)

        return model, cap_pts_ids


def registration(START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, output_dir, seg_fn, fn):
    """
    Registration of surface mesh point set using Elastix
    Performs 3D image registration and move points based on the computed transform
    Cap the surface mesh with test6_2()
    """
    import registration
    import SimpleITK as sitk
    
    # compute volume of all phases to select systole and diastole:
    volume = list()
    # build surface mesh from segmentation at START_PHASE
    fn_poly = os.path.join(output_dir, MODEL_NAME % START_PHASE)
    model, cap_pts_ids = buildSurfaceModelFromImage([seg_fn], fn_poly)
    volume.append(utils.getPolydataVolume(model))
    
    ids = list(range(START_PHASE,TOTAL_PHASE)) + list(range(0,START_PHASE))
    # Only need to register N-1 mesh
    for index in ids[:-1]:
        print("REGISTERING FROM %d TO %d " % (START_PHASE, (index+1)%TOTAL_PHASE))
    
        #ASSUMING increment is 1
        moving_im_fn = os.path.join(image_dir, IMAGE_NAME % ((index+1)%TOTAL_PHASE)) 
        fixed_im_fn =os.path.join(image_dir, IMAGE_NAME % START_PHASE)
        
        fn_out = os.path.join(output_dir, "verts.pts")

        new_model = registration.point_image_transform(utils.resample(sitk.ReadImage(fixed_im_fn)),
            utils.resample(sitk.ReadImage(moving_im_fn)),
            model,
            fn_out
        )
        # Project the cap points so that they are co-planar
        for pt_ids in cap_pts_ids:
            pts = utils.getPolyDataPointCoordinatesFromIDs(new_model, pt_ids)
            new_model = utils.projectOpeningToFitPlane(new_model, pt_ids, pts, 3)
        #ASSUMING increment is 1
        fn_poly = os.path.join(output_dir, MODEL_NAME % ((index+1)%TOTAL_PHASE))
        label_io.writeVTKPolyData(new_model, fn_poly)
        volume.append(utils.getPolydataVolume(new_model))

    SYSTOLE_PHASE = ids[np.argmin(volume)]
    DIASTOLE_PHASE = ids[np.argmax(volume)]
    print("systole, diastole: ", SYSTOLE_PHASE, DIASTOLE_PHASE)
    return DIASTOLE_PHASE

if __name__=="__main__":
    

    PATIENT_ID = 'MACS40282_20150504'
    START_PHASE = 6
    TOTAL_PHASE = 10
    MODEL_NAME = 'phase%d.nii.vtk'
    IMAGE_NAME = 'phase%d.nii'
    SEG_IMAGE_NAME = 'phase%d_pm.nii'
    image_dir = '/Users/fanweikong/Documents/ImageData/4DCCTA/%s/wall_motion_image_volumes' % PATIENT_ID
    output_dir = os.path.join(os.path.dirname(__file__), "meshes_"+PATIENT_ID+"_pm")
    try:
        os.makedirs(os.path.join(output_dir, "surfaces"))
    except Exception as e: print(e)
    try:
        os.makedirs(os.path.join(output_dir, "volumes"))
    except Exception as e: print(e)

    seg_fn = os.path.join('/Users/fanweikong/Documents/ImageData/4DCCTA/', PATIENT_ID, 'wall_motion_labels', SEG_IMAGE_NAME % START_PHASE)
    fn_tempPts = os.path.join(output_dir, "surfaces", 'outputpoints.txt')
    
    #diastole_phase = registration(START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, os.path.join(output_dir, "surfaces"), seg_fn, fn_tempPts)
    volume = []
    for fn in sorted(glob.glob(os.path.join(output_dir, "surfaces", "*.vtk"))):
        poly = label_io.loadVTKMesh(fn)
        volume.append(utils.getPolydataVolume(poly))
    ids = list(range(TOTAL_PHASE))
    ids = sorted([str(i) for i in ids])
    ids = [int(i) for i in ids]
    diastole_phase = ids[np.argmax(volume)]
    systole_phase = ids[np.argmin(volume)]
    print(volume)
    print("diastole, systole: ", diastole_phase, systole_phase)
    diastole_phase = 8
    import subprocess
    path_to_sv = '/Users/fanweikong/SimVascular/build/SimVascular-build/sv'
    fn = os.path.join(output_dir, "surfaces", MODEL_NAME % diastole_phase)
    fn_out = os.path.join(output_dir, "volumes", 'vol_'+ MODEL_NAME % diastole_phase)
    
    subprocess.check_output([path_to_sv, "--python", "--", os.path.join(os.path.dirname(__file__), "sv_main.py"),"--fn", fn, "--fn_out", fn_out])
