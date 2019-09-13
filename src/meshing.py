import os
from sv import *

def meshPolyData(fn, fn_out, args):
    """
    Use SimVascular to mesh a file containing a VTK PolyData and write the volumetric mesh to disk
    Args:
        fn: file name of the VTK PolyData
        fn_out: file name of the output mesh
        args: meshing options, python dic
    Returns:
        None
    """    
    #Set mesh kernel
    MeshObject.SetKernel('TetGen')
    
    #Create mesh object
    msh = MeshObject.pyMeshObject()
    msh.NewObject(fn)
    
    #Load Model
    msh.LoadModel(fn)
    #Create new mesh
    msh.NewMesh()
    for key in args:
        msh.SetMeshOptions(key,[args[key]])
    msh.GenerateMesh()
    #Save mesh to file
    msh.WriteMesh(fn_out)
    if Repository.Exists('ug'):
        Repository.Delete('ug')
    msh.GetUnstructuredGrid('ug')
    Repository.WriteVtkUnstructuredGrid("ug","ascii",fn_out)
