""" 
Functions to plot the mesh surfaces

@author Fanwei Kong
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_surface(verts, faces):
    """
    This function plot a mesh surface

    Args:
        verts: verts output from marching cube algorithm
        faces: faces output from marching cube algorithm
    Returns:
        None
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    plt.tight_layout()
    plt.show()

