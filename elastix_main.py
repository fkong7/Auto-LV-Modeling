import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "src"))

import glob
import numpy as np
import label_io
from models import leftVentricle
from marching_cube import marching_cube, vtk_marching_cube
from image_processing import lvImage
import utils
from registration import Registration
import vtk
import SimpleITK as sitk
import time

def registration(lvmodel, START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, image_dir, output_dir, write=False):
    """
    Registration of surface mesh point set using Elastix
    Performs 3D image registration and move points based on the computed transform
    Cap the surface mesh with test6_2()
    """
    # compute volume of all phases to select systole and diastole:
    volume = list()
    
    volume.append([START_PHASE,lvmodel.getVolume()])

    ids = list(range(START_PHASE,TOTAL_PHASE)) + list(range(0,START_PHASE))

    register = Registration()
    # Only need to register N-1 mesh
    for index in ids[:-1]:
        print("REGISTERING FROM %d TO %d " % (START_PHASE, (index+1)%TOTAL_PHASE))
    
        #ASSUMING increment is 1
        moving_im_fn = os.path.join(image_dir, IMAGE_NAME % ((index+1)%TOTAL_PHASE)) 
        fixed_im_fn =os.path.join(image_dir, IMAGE_NAME % START_PHASE)
        
        register.updateMovingImage(moving_im_fn)
        register.updateFixedImage(fixed_im_fn)

        try:
            os.makedirs(os.path.join(output_dir, "registration"))
        except Exception as e: print(e)
        fn_out = os.path.join(os.path.join(output_dir, "registration"), "verts.pts")

        fn_paras = os.path.join(output_dir, "registration", str(START_PHASE)+"to"+str((index+1)%TOTAL_PHASE)+'.txt')
        new_lvmodel = register.polydata_image_transform(lvmodel, fn_out, fn_paras)
        if write:
            register.writeParameterMap(fn_paras)

        #ASSUMING increment is 1
        fn_poly = os.path.join(output_dir, MODEL_NAME % ((index+1)%TOTAL_PHASE))
        new_lvmodel.writeSurfaceMesh(fn_poly)
        volume.append([(index+1)%TOTAL_PHASE,new_lvmodel.getVolume()])

    np.save(os.path.join(output_dir, "volume.npy"), volume)
    return

if __name__=='__main__':
    start = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--json_fn', help="Name of the json file")
    parser.add_argument('--write', default=False, action='store_true')
    args = parser.parse_args()
    
    paras = label_io.loadJsonArgs(args.json_fn)
    image_dir = os.path.join(paras['im_top_dir'] , paras['patient_id'], paras['im_folder_name'])
    output_dir = os.path.join(paras['out_dir'], paras['patient_id'], "surfaces")
    
    fn_poly = os.path.join(output_dir, paras['model_output'] % paras['start_phase'])

    #
    lvmodel = leftVentricle(label_io.loadVTKMesh(fn_poly))
    registration(lvmodel, paras['start_phase'],paras['total_phase'], paras['model_output'], paras['im_name'], image_dir,output_dir, args.write)
    end = time.time()
    print("Time spent in elastix_main.py: ", end- start)
