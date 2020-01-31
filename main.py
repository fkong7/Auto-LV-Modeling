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
import time

def buildSurfaceModelFromImage(fns, poly_fns, ug_fn=None, remove_ids=[1,4,5,7],la_id=2,aa_id=6):
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
    FACTOR_AA = 1.
    MESH_RESOLUTION = (1.,1.,1.)

    for fn, poly_fn in zip(fns,poly_fns): 

        image = lvImage(fn)
        image.process(remove_ids)

        la_cutter = image.buildCutter(la_id, aa_id, 3, FACTOR_LA, op='valve')
        aa_cutter = image.buildCutter(aa_id, la_id, 3, FACTOR_AA, op='tissue')
        la_fn = '/Users/fanweikong/Downloads/la.vtp'
        label_io.writeVTKPolyData(la_cutter, la_fn)
        aa_fn = '/Users/fanweikong/Downloads/aa.vtp'
        label_io.writeVTKPolyData(aa_cutter, aa_fn)
        image.resample(MESH_RESOLUTION, 'linear')
        image.convert2binary()
        #image.write_image('/Users/fanweikong/Downloads/test.vti')
        model = leftVentricle(image.generate_surface(0, 50))
        #process models
        model.processWall(la_cutter, aa_cutter)
        model.processCap(1.5) 
        fn = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(poly_fn))
        #model.writeSurfaceMesh(fn)
        model.remesh(1.5, fn, poly_fn, ug_fn)
        model.writeSurfaceMesh(poly_fn)


if __name__=="__main__":
    start = time.time()
    from pip._internal import main as pipmain
    pipmain(['install', 'scipy'])
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_fn', nargs=1, help='Name of the json file')
    parser.add_argument('--seg_name', help='Name of the segmentation file')
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
    fn_tempPts = os.path.join(output_dir, "surfaces", 'outputpoints.txt')
    
    #seg_fn = os.path.join(paras['im_top_dir'], paras['patient_id'], paras['seg_folder_name'], paras['seg_name'] % paras['start_phase'])
    #fn_poly = os.path.join(output_dir, "surfaces", paras['model_output'] % paras['start_phase'])
    if args.seg_name is not None:
        seg_fn = os.path.join(paras['im_top_dir'], paras['patient_id'], paras['seg_folder_name'], args.seg_name)
        fn_poly = os.path.join(output_dir, "surfaces", args.seg_name+'.vtk')
    else:
        seg_fn = os.path.join(paras['im_top_dir'], paras['patient_id'], paras['seg_folder_name'], paras['seg_name'])
        fn_poly = os.path.join(output_dir, "surfaces", paras['model_output'])

    print(seg_fn, fn_poly)
    #run volume mesh to generate ids but not using it
    fn_ug = 'temp'
    buildSurfaceModelFromImage([seg_fn], [fn_poly], fn_ug)

    end = time.time()
    print("Time spend in main.py: ", end-start)
