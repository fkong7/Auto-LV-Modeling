import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "src"))

import glob
import numpy as np
import label_io
from image_processing import lvImage
from models import leftVentricle
from marching_cube import marching_cube, vtk_marching_cube
import utils
import vtk



def buildSurfaceModelFromImage(fns, poly_fn, ug_fn=None):
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
    FACTOR_LA = 0.7
    FACTOR_AA = 1.2
    MESH_RESOLUTION = (1.5,1.5,1.5)

    for fn in fns: 

        image = lvImage(fn)
        image.process([1,4,5,7])

        la_cutter = image.buildCutter(2, 3, FACTOR_LA, op='valve')
        aa_cutter = image.buildCutter(6, 3, FACTOR_AA, op='tissue')
        image.convert2binary()
        image.resample(MESH_RESOLUTION, 'linear')
        model = leftVentricle(image.generate_surface(0, 50))
        #process models
        model.processWall(la_cutter, aa_cutter)
        model.processCap(1.5) 
        fn = os.path.join(os.path.dirname(__file__), "debug", "temp.vtk")
        model.writeSurfaceMesh(fn)
        model.remesh(1.5, fn, poly_fn, ug_fn)
        model.writeSurfaceMesh(poly_fn)
        return model


    
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

    from pip._internal import main as pipmain
    pipmain(['install', 'scipy'])
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_fn', nargs=1, help='Name of the json file')
    args = parser.parse_args()
    
    paras = label_io.loadJsonArgs(args.json_fn[0])

    image_dir = os.path.join(paras['im_top_dir'] , paras['patient_id'], paras['im_folder_name'])
    output_dir = os.path.join(paras['out_dir'], paras['patient_id'])
    try:
        os.makedirs(os.path.join(output_dir, "surfaces"))
    except Exception as e: print(e)
    try:
        os.makedirs(os.path.join(output_dir, "volumes"))
    except Exception as e: print(e)
    
    seg_fn = os.path.join(paras['im_top_dir'], paras['patient_id'], paras['seg_folder_name'], paras['seg_name'] % paras['start_phase'])
    fn_tempPts = os.path.join(output_dir, "surfaces", 'outputpoints.txt')
    
    fn_poly = os.path.join(output_dir, "surfaces", paras['model_output'] % paras['start_phase'])
    model = buildSurfaceModelFromImage([seg_fn], fn_poly)


