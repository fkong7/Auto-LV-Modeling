import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "src"))

import numpy as np
import label_io
from models import leftVentricle
from registration import Registration
import time

def registration(lvmodel, START_PHASE, TOTAL_PHASE, MODEL_NAME, IMAGE_NAME, image_dir, surface_dir, output_dir, mask_dir, write=False, smooth=False):
    """
    Registration of surface mesh point set using Elastix
    Performs 3D image registration and move points based on the computed transform
    Cap the surface mesh with test6_2()
    """
    # compute volume of all phases to select systole and diastole:

    ids = list(range(START_PHASE,TOTAL_PHASE)) + list(range(0,START_PHASE))
    #ids = [9, START_PHASE]
    reg_output_dir = os.path.join(output_dir, "registration")
    try:
        os.makedirs(reg_output_dir)
    except Exception as e: print(e)

    register = Registration(smooth)
    # Only need to register N-1 mesh
    fixed_im_fn =os.path.join(image_dir, IMAGE_NAME % START_PHASE)
    fixed_mask_fn =os.path.join(mask_dir, IMAGE_NAME % START_PHASE)
    fn_poly = os.path.join(output_dir, MODEL_NAME % START_PHASE)
    lvmodel.writeSurfaceMesh(fn_poly)
    volume = list()
    volume.append([START_PHASE,lvmodel.getVolume()])
    image_output_dir = os.path.join(reg_output_dir, "images")
    for index in ids[:-1]:
        print("REGISTERING FROM %d TO %d " % (START_PHASE, (index+1)%TOTAL_PHASE))
    
        #ASSUMING increment is 1
        moving_im_fn = os.path.join(image_dir, IMAGE_NAME % ((index+1)%TOTAL_PHASE)) 
        #moving_mask_fn = os.path.join(mask_dir, IMAGE_NAME % ((index+1)%TOTAL_PHASE)) 
        
        register.updateMovingImage(moving_im_fn)
        #register.updateMovingMask(moving_mask_fn)
        register.updateFixedImage(fixed_im_fn)
        register.updateFixedMask(fixed_mask_fn)

        try:
            os.makedirs(os.path.join(image_output_dir))
        except Exception as e: print(e)
        fn_out = os.path.join(os.path.join(reg_output_dir), "verts.pts")

        fn_paras = os.path.join(reg_output_dir, str(START_PHASE)+"to"+str((index+1)%TOTAL_PHASE)+'.txt')
        new_lvmodel = register.polydata_image_transform(lvmodel, fn_out, os.path.join(image_output_dir, IMAGE_NAME % ((index+1)%TOTAL_PHASE)) , fn_paras)
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
    
    parser.add_argument('--image_dir', help='Path to the ct/mr images or segmentation results')
    parser.add_argument('--mask_dir', help='Path to the mask file')
    parser.add_argument('--surface_dir', help='Path to the unregistered surface mesh')
    parser.add_argument('--output_dir', help='Path to the registered surface meshes')
    parser.add_argument('--start_phase', type=int, help='Phase ID of the surface mesh used as the registration target')
    parser.add_argument('--total_phase', type=int, help='Total number of phases')
    parser.add_argument('--edge_size', type=float, help='Maximum edge size of the surface mesh')
    parser.add_argument('--model_output', help='Output format of registered surfaces')
    parser.add_argument('--im_name', help='Name of the images in image_dir')
    parser.add_argument('--write', default=False, action='store_true')
    parser.add_argument('--smooth', default=False, action='store_true')
    args = parser.parse_args()
    
    image_dir = args.image_dir
    surface_dir = args.surface_dir
    output_dir = args.output_dir
    mask_dir = args.mask_dir
    fn_poly = os.path.join(surface_dir, args.model_output % args.start_phase)

    #
    lvmodel = leftVentricle(label_io.loadVTKMesh(fn_poly), edge_size=args.edge_size )
    registration(lvmodel, args.start_phase,args.total_phase, args.model_output, args.im_name, image_dir,surface_dir, output_dir, mask_dir, args.write, args.smooth)
    end = time.time()
    print("Time spent in elastix_main.py: ", end-start)
