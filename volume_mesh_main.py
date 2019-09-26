import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))
import argparse
import numpy as np
import meshing
import models
import label_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--json_fn', help="Name of the json file")
    parser.add_argument('--phase', default=-1, type=int, help="Id of the phase to generate volume mesh")
    args = parser.parse_args()

    paras = label_io.loadJsonArgs(args.json_fn)
    
    output_dir = os.path.join(paras['out_dir'], paras['patient_id'], "surfaces")

    volume_fn = np.load(os.path.join(output_dir, "volume.npy"))
    if args.phase == -1:
        phase = int(np.argmax(volume_fn))
    else:
        phase = args.phase
    poly_fn = os.path.join(output_dir, paras['model_output'] % phase)

    lvmodel = models.leftVentricle(label_io.loadVTKMesh(poly_fn))
    
    output_vol = os.path.join(paras['out_dir'], paras['patient_id'],"volumes", "vol_phase%d.vtu"%phase)
    lvmodel.remesh(2., poly_fn, poly_fn=None, ug_fn=output_vol)
    lvmodel.writeVolumeMesh(output_vol)

