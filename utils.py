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

def eraseBoundary(labels, pixels, bg_id):
    """
    Erase anything on the boundary by a specified number of pixels

    Args:
        labels: python nd array 
        pixels: number of pixel width to erase
        bg_id: id number of background class
    Returns:
        labels: editted label maps
    """
    x,y,z = labels.shape
    labels[:pixels,:,:] = bg_id
    labels[-pixels:,:,:] = bg_id
    labels[:,:pixels,:] = bg_id
    labels[:,-pixels:,:] = bg_id
    labels[:,:,:pixels] = bg_id
    labels[:,:,-pixels:] = bg_id
    return labels

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

def recolorPixelsByPlane(labels, ori, nrm, bg_id):
    """
    For every pixel above a plane in physcal coordinates, change the pixel value to background pixel value

    Args:
        labels: SimpleItk image
        ori: plane origin
        nrm: plane normal
    Returns:
        labels: editted SimpleItk image
    """
    def isAbovePlane(pt1, ori, nrm):
        vec1 = np.array(pt1)-np.array(ori)
        vec2 = np.array(nrm)
        dot = np.dot(vec1, vec2)
        if dot>0:
            return True
        else:
            return False

    X, Y, Z = labels.GetSize()
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                coords = labels.TransformIndexToPhysicalPoint((x,y,z))
                if isAbovePlane(coords, ori, nrm):
                    labels.SetPixel(x, y, z, bg_id)

    return labels


    
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

def labelDilateErode(im, label_id, bg_id,thickness):
    """
    Dilates a label to create thickness 
    
    Args:
        im: vtkImage of the label map
        label_id: class id to erode
        bg_id: class id of backgroud to dilate
        thickness: thickness of the erosion in physical unit
    Returns
        newIm: vtkImage with thickened boundary of the tissue structure
    """
    dilateErode = vtk.vtkImageDilateErode3D()
    dilateErode.SetInputData(im)
    dilateErode.SetDilateValue(label_id)
    dilateErode.SetErodeValue(bg_id)
    
    kernel_size = np.rint(thickness/np.array(im.GetSpacing())).astype(int)
    print(kernel_size)
    dilateErode.SetKernelSize(*kernel_size)
    dilateErode.Update()
    newIm = dilateErode.GetOutput()
    
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    pyIm_new = vtk_to_numpy(newIm.GetPointData().GetScalars())
    pyIm = vtk_to_numpy(im.GetPointData().GetScalars())
    pyIm_new[np.where((pyIm_new-pyIm==0)&(pyIm==label_id))]=bg_id
   # convert to binary
    pyIm_new[np.where(pyIm_new!=0)] = 1 
    newIm.GetPointData().SetScalars(numpy_to_vtk(pyIm_new))
    return newIm

def clipVTKPolyData(poly, ori, nrm):
    """
    Clip a VTK PolyData with a plane by specifying the plane normal and origin
    TO-DO: future improvements to close the cut: https://github.com/Kitware/VTK/blob/master/Examples/VisualizationAlgorithms/Python/ClipCow.py

    Args:
        poly: VTK PolyData
        ori: plane origin, tuple
        nrm: plane normal, tuple
    Returns:
        poly: clipped VTK PolyData
    """
    polyNormals = vtk.vtkPolyDataNormals()
    polyNormals.SetInputData(poly)

    plane = vtk.vtkPlane()
    plane.SetOrigin(*ori)
    plane.SetNormal(*nrm)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetClipFunction(plane)
    clipper.GenerateClipScalarsOn()
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(0.5)
    clipper.Update()
    
    cutEdges = vtk.vtkCutter()
    cutEdges.SetInputConnection(polyNormals.GetOutputPort())
    cutEdges.SetCutFunction(plane)
    cutEdges.GenerateCutScalarsOn()
    cutEdges.SetValue(0, 0.5)
    cutEdges.Update()

    cutStrips = vtk.vtkStripper()
    cutStrips.SetInputConnection(cutEdges.GetOutputPort())
    cutStrips.Update()
    cutPoly = vtk.vtkPolyData()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())
    
    cutTriangles = vtk.vtkTriangleFilter()
    cutTriangles.SetInputData(cutPoly)
    cutTriangles.Update()
    poly = appendVTKPolydata(clipper.GetOutput(), cutTriangles.GetOutput())

    return poly

def recolorVTKPixelsByPlane(labels, ori, nrm, bg_id):
    """
    For every pixel above a plane in physcal coordinates, change the pixel value to background pixel value

    Args:
        labels: VTK image
        ori: plane origin
        nrm: plane normal
    Returns:
        labels: editted VTK image
    """

    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    pyLabel = vtk_to_numpy(labels.GetPointData().GetScalars())
    spacing = labels.GetSpacing()
    origin = labels.GetOrigin()
    X, Y, Z = labels.GetDimensions()
    total_num = X*Y*Z
    
    x, y, z = np.meshgrid(range(X), range(Y), range(Z))
    indices = np.moveaxis(np.vstack((z.flatten(),y.flatten(),x.flatten())),0,1)
    b = np.tile(spacing, total_num).reshape(total_num,3)
    physical = indices * b +np.tile(origin, total_num).reshape(total_num,3)
    vec1 = physical - np.tile(ori, total_num).reshape(total_num,3)
    vec2 = np.tile(nrm, total_num).reshape(total_num,3)
    below = np.sum(vec1*vec2, axis=1)<0
    pyLabel[below] = bg_id
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel))

    return labels

def vtkImageResample(image, dims, opt):
    """
    Resamples the vtk image to the given dimenstion

    Args:
        image: vtk Image data
        dims: image dimension
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """

    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt=='linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt=='NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt=='cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*new_spacing)
    reslicer.Update()

    return reslicer.GetOutput()
