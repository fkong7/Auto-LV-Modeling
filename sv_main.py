import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))
import argparse
import meshing
import label_io
"""
    Functions needing the SV Python interpreter
"""
def volumetric_mesh(fn, fn_out):
    """
    Example to generate volumetric mesh from surface mesh
    """
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

def surface_and_volumetric_mesh(fn, fn_out):
    """
    Remesh surface mesh
    """
    mesh_ops = {
            'SurfaceMeshFlag': True,
            'VolumeMeshFlag': True,
            'GlobalEdgeSize': 2., 
            'MeshWallFirst': True, 
            'NoMerge':True,
            'NoBisect': True,
            'Epsilon': 1e-8,
            'Optimization': 3,
            'QualityRatio': 1.4
    }
    meshing.meshPolyData(fn, fn_out, mesh_ops)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', nargs=1, help='Name of the VTK PolyData file')
    parser.add_argument('--fn_cap', nargs=1, help='Name of the capped VTK PolyData to write')
    parser.add_argument('--fn_out', nargs=1, help='Name of the meshed fils to write')
    args = parser.parse_args()
    #volumetric_mesh(args.fn[0], args.fn_out[0])
    poly = label_io.loadVTKMesh(args.fn[0])
    ids = meshing.capPolyDataWithIds(poly, args.fn[0], args.fn_cap[0], 1, False)
    remeshed_fn = os.path.splitext(args.fn_cap[0])[0]+'_remeshed.vtk'
    meshing.remeshPolyData(args.fn_cap[0], remeshed_fn, 0.9, 1.1, True)
    surface_and_volumetric_mesh(remeshed_fn, args.fn_out[0]) 

