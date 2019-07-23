"""
Use marching cube algorithm to create iso-surface of label map

@author Fanwei Kong
"""
from skimage import measure

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



