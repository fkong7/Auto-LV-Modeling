import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))
import argparse
import numpy as np
import meshing
import models
import io_utils
import time

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', help="Path to the surface meshes")
    parser.add_argument('--output_dir', help="Path to the volume meshes")
    parser.add_argument('--model_out', help="Name format of surface")
    parser.add_argument('--edge_size', type=float, help="Name format of surface")
    parser.add_argument('--phase', default=-1, type=int, help="Id of the phase to generate volume mesh")
    args = parser.parse_args()
    
    input_dir = args.input_dir

    if args.phase == -1:
        try:
            volume_fn = np.load(os.path.join(input_dir, "volume.npy"))
            phase = volume_fn[:,0][int(np.argmax(volume_fn[:,1]))]
        except:
            print("Mesh volumes not found, the first model will be meshed")
            phase = 0
    else:
        phase = args.phase
    poly_fn = os.path.join(input_dir, args.model_out % phase)

    lvmodel = models.LeftVentricle(io_utils.read_vtk_mesh(poly_fn))
    
    output_vol = os.path.join(args.output_dir, 'mesh-complete')
    lvmodel.remesh(args.edge_size, poly_fn, poly_fn=None, ug_fn=output_vol, mmg=False)
    lvmodel.write_mesh_complete(output_vol)
    end = time.time()
    print("Time spent in volume_mesh_main.py: ", end-start)
    print("Mesh generated for ", poly_fn)
