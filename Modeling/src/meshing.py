import os
from sv import *

def meshPolyData(fn, fns_out, args):
    """
    Use SimVascular to mesh a file containing a VTK PolyData and write the volumetric mesh to disk
    Args:
        fn: file name of the VTK PolyData
        fns_out: file names of the output mesh (poly_fn, ug_fn)
        args: meshing options, python dic
    Returns:
        None
    """    
    #Set mesh kernel
    MeshObject.SetKernel('TetGen')
    
    #Create mesh object
    msh = MeshObject.pyMeshObject()
    if Repository.Exists(fn+'mesh'):
        msh.GetMesh(fn+'mesh')
    else:
        msh.NewObject(fn+'mesh')
   
    #Load Model
    msh.LoadModel(fn)
    
    #Create new mesh
    msh.NewMesh()
    msh.SetWalls([1])
    
    for key in args:
        msh.SetMeshOptions(key,[args[key]])

   # msh.SetSizeFunctionBasedMesh(args['MeshSizingFunction'],'MeshSizingFunction')
    msh.GenerateMesh()
    #Save mesh to file
    #msh.WriteMesh(fns_out[0])
   
    poly_fn, ug_fn = fns_out
    if args['SurfaceMeshFlag']:
        msh.GetPolyData(poly_fn)
    if args['VolumeMeshFlag']:
        msh.GetUnstructuredGrid(ug_fn)
    return (poly_fn, ug_fn)

def capPolyDataWithIds(poly, poly_name, capped_name, start=1, write=False, remesh=True):
    """
    Use SimVascular to cap a surface mesh with IDs

    Args:
        poly: VTK PolyData
        poly_name: name to store the VTK PolyData in repository
        capped_name: name to store the output in repository
        start: smallest id to assign
    Returns:
        None
    """
    if Repository.Exists(poly_name):
        Repository.Delete(poly_name)
    Repository.ImportVtkPd(poly, poly_name)

    ids = VMTKUtils.Cap_with_ids(poly_name, capped_name, start, 2) 
    Repository.Delete(poly_name)
    if write:
        Repository.WriteVtkPolyData(capped_name, 'ascii', capped_name)
    return

def remeshPolyData(poly_name, remeshed_name, hmin, hmax,write=False):
    """
    Use SimVascular MMG remesh to remesh a surfac mesh
    
    Args:
        poly_name: name of the polydata in repository
        remeshed_name: name of the remeshed polydata in reporitory
        hmin: min edge size
        hmax: max edge size
        write: if write the remeshed surface to disk
    Returns:
        None
    """

    if not Repository.Exists(poly_name):
        raise AttributeError("%s not in repository" % poly_name)
        return

    MeshUtil.Remesh(poly_name, remeshed_name, hmin, hmax)
    if write is not None:
        Repository.WriteVtkPolyData(remeshed_name, 'ascii', write)
    return

