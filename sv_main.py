import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src"))
"""
    Functions needing the SV Python interpreter
"""
def mesh():
    """
    Example to generate volumetric mesh from surface mesh
    """
    fn = os.path.join(os.path.dirname(__file__), "4dct_model", "phase7.nii.vtk")
    import meshing
    fn_out = os.path.join(os.path.dirname(__file__), "4dct_model", "phase7_vol.vtk")
    # generate volumetric mesh:
    mesh_ops = {
            'SurfaceMeshFlag': False,
            'VolumeMeshFlag': True,
            'GlobalEdgeSize': 2.
    }
    meshing.meshPolyData(fn, fn_out, mesh_ops)

if __name__=="__main__":
    mesh()
