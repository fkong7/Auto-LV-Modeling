"""
Utility functions for label map editing

@author Fanwei Kong
"""
import numpy as np
import SimpleITK as sitk
import vtk

########################
## Label Map functions
########################
def convert2binary(labels):
    """
    This function converts a Simple ITK label to binary label
    
    Args:
        labels: SimpleITK image
    Returns:
        pyLabel: ndnumpy array
    """
    pyLabel = sitk.GetArrayFromImage(labels)
    pyLabel[np.where(pyLabel!=0)] = 1
    
    #bnryLabel = sitk.GetImageFromArray(pyLabel)
    

    return pyLabel

def removeClass(labels, class_id, bg_id):
    """
    Convert class label to background label

    Args:
        class_id: the id number of the class to remove
        labels: label map
        bg_id: id number of background
    Returns:
        labels: edited label map
    """
    labels[np.where(labels==class_id)] = bg_id
    return labels

def gaussianSmoothImage(im, stdev):
    """
    Smooths a python ndarray Image with Gaussian smoothing

    Args:
        im: Python nd array
        stdev: standard deviation for Gaussian smoothing
    Returns:
        im: smoothed Image
    """
    from scipy.ndimage.filters import gaussian_filter
    im = gaussian_filter(im, stdev)

    return im


################################
## VTK PolyData functions
###############################
def appendVTKPolydata(poly1, poly2):
    """ 
    This function appends two polydata

    Args:
        poly1: first vtk polydata
        poly2: second vtk polydata
    Returns:
        poly: appended polydata
    """

    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(poly1)
    appender.AddInputData(poly2)
    appender.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(appender.GetOutputPort())
    cleaner.Update()

    poly = cleaner.GetOutput()
    return poly

def smoothVTKPolydata(poly, iteration=100):
    """
    This function smooths a vtk polydata

    Args:
        poly: vtk polydata to smooth

    Returns:
        smoothed: smoothed vtk polydata
    """

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed

def booleanVTKPolyData(poly1, poly2, keyword):
    """
    Apply VTK boolean operation on two VTK PolyData

    Args:
        poly1: first VTK PolyData
        poly2: second VTK PolyData
        keywords: str union, intersection, difference
    Returns:
        poly: resulted VTK PolyData
    """

    boolean = vtk.vtkBooleanOperationPolyDataFilter()
    if keyword=="union":
        boolean.SetOperationToUnion()
    elif keyword=="intersection":
        boolean.SetOperationToIntersection()
    elif keyword=="difference":
        boolean.SetOperationToDifference()
    else:
        raise ValueError("Keyword option is not supporte.")

    boolean.SetInputData(0, poly1)
    boolean.SetInputData(1, poly2)
    boolean.Update()

    return boolean.GetOuptut()

def fillHole(poly):
    """
    Fill holes in VTK PolyData
    
    Args:
        poly: VTK PolyData to fill
    Returns:
        poly: filled VTK PolyData
    """
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(poly)
    filler.SetHoleSize(100000000.)
    filler.Update()
    
    return filler.GetOutput()

def setCellScalar(poly, scalar):
    """
    Assign a scalar value to each cell of the polydata

    Args:
        poly; VTK PolyData to edit
        scalar: scalar value float
    Returns:
        poly: edited VTK PolyData
    """
    num = poly.GetNumberOfCells()
    cellData = vtk.vtkFloatArray()
    cellData.SetNumberOfValues(num)
    for i in range(num):
        cellData.SetValue(i, scalar)
    poly.GetCellData().SetScalars(cellData)
    
    return poly

def gaussianSmoothVTKImage(im, stdev):
    """
    Smooths a vtk Image with Gaussian smoothing

    Args:
        im: vtkImage
        stdev: standard deviation for Gaussian smoothing
    Returns:
        im: smoothed vtkImage
    """
    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputData(im)
    print(im.GetSpacing())
    smoother.SetRadiusFactors(np.array(im.GetSpacing())*stdev)
    smoother.Update()
    return smoother.GetOutput()
