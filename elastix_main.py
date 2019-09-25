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
import registration
import vtk
import SimpleITK as sitk

def registration(poly_fn, START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, output_dir):
    """
    Registration of surface mesh point set using Elastix
    Performs 3D image registration and move points based on the computed transform
    Cap the surface mesh with test6_2()
    """
    # compute volume of all phases to select systole and diastole:
    volume = list()
    
    lvmodel = leftVentricle(label_io.loadVTKMesh(poly_fn))
    volume.append(lvmodel.getVolume())

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
        register.computeTransform()

        fn_out = os.path.join(output_dir, "verts.pts")
        new_lvmodel = register.polydata_image_transform(lvmodel, fn_out) 

        #ASSUMING increment is 1
        fn_poly = os.path.join(output_dir, MODEL_NAME % ((index+1)%TOTAL_PHASE))
        new_lvmodel.writeSurfaceMesh(fn_poly)
        volume.append(new_lvmodel.getVolume())

    SYSTOLE_PHASE = ids[np.argmin(volume)]
    DIASTOLE_PHASE = ids[np.argmax(volume)]
    print("systole, diastole: ", SYSTOLE_PHASE, DIASTOLE_PHASE)
    return DIASTOLE_PHASE

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_fn', nargs=1, help='Name of the json file')
    args = parser.parse_args()

    import json

    with open(json_fn) as data_file:
        data = json.load(data_file)

     
    diasole = registration(poly_fn, START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, output_dir)
