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



def buildSurfaceModelFromImage(fns, fns_out):
    """
    Modified test6 to cut on the PolyData directly to create better defined inlet/outlet geometry
    The left atrium is cut normal to the direction defined by the normal of the mitral plane
    The amount of left atrium kept can be adjusted by a scalar factor, 
    which scales the distance between mv plane centroid and la centroid

    Args:
        fns: list containing the paths to images
        fns_out: output file names (poly_fn, ug_fn)
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
        model = leftVentricle(m_c.vtk_marching_cube_multi(vtkIm, 0, 50))
        model.processWall(la_cutter, aa_cutter)
        model.processCap(1.5) 
        fn = os.pathjoin(os.path.dirname(__file__), "debug", "temp.vtk")
        model.writeSurfaceMesh(fn)
        model.remesh(2., fn, fns_out)
        model.writeSurfaceMesh(fns_out[0])
        model.writeVolumeMesh(fns_out[1])
        return model


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

    
def test1():
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

if __name__=="__main__":
    PATIENT_ID = 'MACS40282_20150504'
    START_PHASE = 6
    TOTAL_PHASE = 10
    MODEL_NAME = 'phase%d.nii.vtk'
    IMAGE_NAME = 'phase%d.nii'
    SEG_IMAGE_NAME = 'phase%d_pm.nii'
    image_dir = '/Users/fanweikong/Documents/ImageData/4DCCTA/%s/wall_motion_image_volumes' % PATIENT_ID
    output_dir = os.path.join(os.path.dirname(__file__), "debug")
    try:
        os.makedirs(os.path.join(output_dir, "surfaces"))
    except Exception as e: print(e)
    try:
        os.makedirs(os.path.join(output_dir, "volumes"))
    except Exception as e: print(e)

    seg_fn = os.path.join('/Users/fanweikong/Documents/ImageData/4DCCTA/', PATIENT_ID, 'wall_motion_labels', SEG_IMAGE_NAME % START_PHASE)
    fn_tempPts = os.path.join(output_dir, "surfaces", 'outputpoints.txt')
    
    fn_poly = os.path.join(output_dir, MODEL_NAME % START_PHASE)
    fn_ug = os.path.join(output_dir, "volumes", 'vol_'+ MODEL_NAME % diastole_phase)
    model = buildSurfaceModelFromImage([seg_fn], (fn_poly, fn_ug))
    #diastole_phase = registration(START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, os.path.join(output_dir, "surfaces"), seg_fn, fn_tempPts)
