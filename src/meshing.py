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
    msh.NewObject(fn+'mesh')
    
    #Load Model
    msh.LoadModel(fn)
    #Create new mesh
    msh.NewMesh()
    msh.SetWalls([1,2,3])
    print("GetModelFaceInfo")
    print(msh.GetModelFaceInfo())
    
    for key in args:
        msh.SetMeshOptions(key,[args[key]])

   # msh.SetSizeFunctionBasedMesh(args['MeshSizingFunction'],'MeshSizingFunction')
    msh.GenerateMesh()
    #Save mesh to file
    msh.WriteMesh(fn_out)
   
    if args['SurfaceMeshFlag']:
        poly_fn = os.path.splitext(fn_out)[0]+'_surface.vtk'
        if Repository.Exists(poly_fn):
            Repository.Delete(poly_fn)
        msh.GetPolyData(poly_fn)
    Repository.WriteVtkPolyData(poly_fn, "ascii", poly_fn)
    if args['VolumeMeshFlag']:
        ug_fn = os.path.splitext(fn_out)[0]+'_volume.vtk'
        if Repository.Exists(ug_fn):
            Repository.Delete(ug_fn)
        msh.GetUnstructuredGrid(ug_fn)
        Repository.WriteVtkUnstructuredGrid(ug_fn,"ascii",ug_fn)

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
    if write:
        Repository.WriteVtkPolyData(remeshed_name, 'ascii', remeshed_name)
    return

