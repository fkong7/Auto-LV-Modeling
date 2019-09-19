import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))
import argparse

"""
    Functions needing the SV Python interpreter
"""
def mesh(fn, fn_out):
    """
    Example to generate volumetric mesh from surface mesh
    """
    import meshing
    # generate volumetric mesh:
    mesh_ops = {
            'SurfaceMeshFlag': False,
            'VolumeMeshFlag': True,
            'GlobalEdgeSize': 2., 
            'NoMerge':True,
            'NoBisect': True,
            'Optimization': 3,
            'QualityRatio': 1.4
    }
    meshing.meshPolyData(fn, fn_out, mesh_ops)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', nargs=1, help='Name of the VTK PolyData file')
    parser.add_argument('--fn_out', nargs=1, help='Name of the volumertic file to write')
    args = parser.parse_args()
    mesh(args.fn[0], args.fn_out[0])
