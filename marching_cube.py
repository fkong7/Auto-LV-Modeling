"""
Use marching cube algorithm to create iso-surface of label map

@author Fanwei Kong
"""
from skimage import measure
import vtk
import utils

def marching_cube(label, tol):
    """
    Args:
        label: numpy array of label map
        tol: threshold value for iso-surface
    Returns
        mesh: tuple containing outputs of marching cube algorithm
    """

    verts, faces, normals, values = measure.marching_cubes_lewiner(label, tol)
    
    return (verts, faces, normals, values)

def vtk_marching_cube(vtkLabel, tol, smooth=True):
    """
    Use the VTK marching cube implementation to create the surface mesh

    Args:
        vtkLabel: vtk structured array containing the label map
        tol: threshold value for iso-surface
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, tol)
    contour.Update()

    mesh = contour.GetOutput()

    if smooth:
        mesh = utils.smoothVTKPolydata(mesh)

    return mesh
