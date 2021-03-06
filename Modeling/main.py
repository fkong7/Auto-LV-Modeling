import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "src"))

import glob
import numpy as np
import label_io
from image_processing import lvImage
from models import leftVentricle, leftHeart
from marching_cube import marching_cube, vtk_marching_cube
import utils
import vtk
import time

def buildLVModelFromImage(fns, poly_fns, ug_fn=None, remove_ids=[1,4,5,7],la_id=2,aa_id=6, edge_size = 1., timming=False, use_SV=True):
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
    FACTOR_LA = 18
    FACTOR_AA = 38
    MESH_RESOLUTION = (1.,1.,1.)
        
        
    time_list = []
    if timming:
        start = time.time()
    for fn, poly_fn in zip(fns,poly_fns): 

        image = lvImage(fn)
        image.process(remove_ids)

        la_cutter = image.buildCutter(la_id, aa_id, 3, FACTOR_LA, op='valve')
        aa_cutter = image.buildCutter(aa_id, la_id, 3, FACTOR_AA, op='tissue')
        image.resample(MESH_RESOLUTION, 'linear')
        image.convert2binary()

        if timming:
            im_time = time.time() - start
            time_now = time.time()
        
        model = leftVentricle(image.generate_surface(0, smooth_iter=20, band=0.02))
        #process models
        model.processWall(*la_cutter, *aa_cutter)
        model.processCap(5.) 
        if timming:
            surf_time = time.time() - time_now
            time_now = time.time()
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "debug"))
        except Exception as e: print(e)
        fn = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(poly_fn))
        if use_SV:
            model.remesh(edge_size, fn, poly_fn, ug_fn)
        model.writeSurfaceMesh(poly_fn)
        if timming:
            mesh_time = time.time() - time_now
            time_list.append([im_time, surf_time, mesh_time])
    return time_list

def buildLeftHeartModelFromImage(fns, poly_fns, ug_fn=None, remove_ids=[1,4,5,7], la_id=2, aa_id=6, edge_size = 1., timming=False, use_SV=True):
    
    MESH_RESOLUTION = (0.5,0.5,0.5)
    FACTOR_AA = 38
        
    time_list = []
    if timming:
        start = time.time()
    for fn, poly_fn in zip(fns,poly_fns): 

        image = lvImage(fn)
        image.process(remove_ids)

        aa_cutter = image.buildCutter(aa_id, la_id, 3, FACTOR_AA, op='tissue')
        image.resample(MESH_RESOLUTION, 'linear')
        image.convert2binary()
        image.erase_boundary()

        if timming:
            im_time = time.time() - start
            time_now = time.time()
        
        model = leftHeart(image.generate_surface(0, smooth_iter=20, band=0.02))
        model.processWall(*aa_cutter)
        model.processCap(5.) 
        if timming:
            surf_time = time.time() - time_now
            time_now = time.time()
        fn = os.path.join(os.path.dirname(__file__), "debug", os.path.basename(poly_fn))
        if use_SV:
            model.remesh(edge_size, fn, poly_fn, ug_fn)
        model.writeSurfaceMesh(poly_fn)
        if timming:
            mesh_time = time.time() - time_now
            time_list.append([im_time, surf_time, mesh_time])
    return time_list

if __name__=="__main__":
    start = time.time()
    #from pip._internal import main as pipmain
    #pipmain(['install', 'scipy'])
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_SV',default=True, action='store_false', help='Whether to disable SV for remeshing')
    parser.add_argument('--seg_name', help='Name of the segmentation file')
    parser.add_argument('--input_dir', help='Path to the segmentation directory')
    parser.add_argument('--output_dir', help='Path to the output directory')
    parser.add_argument('--edge_size', type=float, help='Maximum edge size of the surface mesh')
    args = parser.parse_args()
    
    try:
        os.makedirs(os.path.join(args.output_dir))
    except Exception as e: print(e)
    # try:
    #     os.makedirs(os.path.join(args.output_dir, "volumes"))
    # except Exception as e: print(e)
    # fn_tempPts = os.path.join(args.output_dir, "surfaces", 'outputpoints.txt')
    
    seg_fn = os.path.join(args.input_dir, args.seg_name)
    print(seg_fn)
    fn_poly = os.path.join(args.output_dir, args.seg_name+'.vtp')
    
    #run volume mesh to generate ids but do not use it
    fn_ug = 'temp'
    timming = True
    time_list = buildLVModelFromImage([seg_fn], [fn_poly], fn_ug, edge_size=args.edge_size, timming=timming, use_SV=args.disable_SV)
    if timming:
        import csv
        with open(os.path.join(args.output_dir, 'time_results.csv'), 'a' , newline="") as f:
            writer = csv.writer(f)
            writer.writerows(time_list)

    end = time.time()
    print("Time spend in Modeling/main.py: ", end-start)
