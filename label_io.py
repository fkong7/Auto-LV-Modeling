"""
IO functions for importing and exporting label maps and mesh surfaces

@author: Fanwei Kong

"""
import SimpleITK as sitk
import numpy as np
import os

def loadLabelMap(fn):
    """ 
    This function import the label map as numpy arrays.

    Args: 
        fn: filename of the label map

    Return:
        label: numpy array of the label map
        spacing: spacing information of the label map
    """
    mask = sitk.ReadImage(fn)
    label = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()

    return label, spacing


