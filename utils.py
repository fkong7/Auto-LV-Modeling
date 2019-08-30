"""
Utility functions for label map editing

@author Fanwei Kong
"""
import numpy as np
import SimpleITK as sitk
import vtk
import scipy

##########################
## Numpy Utility functions
##########################

def fitPlaneNormal(points_input):
    """
    Fit a plane to a point set
    
    Args:
        points_input: 3d coordinates of a point set
    Returns:
        normal: normal of the fitting plane
    """
    import functools
    def _cross(a, b):
        """
        Cross product of two vectors
        """
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

    def _plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a*x + b*y + c
        return z
    
    def _error(params, points):
        result = 0
        for (x,y,z) in points:
            plane_z = _plane(x, y, params)
            diff = abs(plane_z - z)
            result += diff**2
        return result
    fun = functools.partial(_error, points=points_input)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)
    a = res.x[0]
    b = res.x[1]
    c = res.x[2]
    point  = np.array([0.0, 0.0, c])
    normal = np.array(_cross([1,0,a], [0,1,b]))
    D = -point.dot(normal)
    return normal

########################
## Label Map functions
########################
 
def resample(image, resolution = (0.5, 0.5, 0.5), dim=3):
  """
  This function resamples a SimpleITK image to desired resolution

  Args:
    image: SimpleItk image
    resolution: desired grid resolution
    dim: image dimension
  Returns:
    newimage: resampeled SimpleITK image
  """
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(sitk.sitkNearestNeighbor)
  resample.SetOutputDirection(image.GetDirection())
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputSpacing(resolution)

  orig_size = np.array(image.GetSize(), dtype=np.int)
  orig_spacing = np.array(image.GetSpacing())
  new_size = orig_size*(orig_spacing/np.array(resolution))
  new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
  new_size = [int(s) for s in new_size]
  resample.SetSize(new_size)
  newimage = resample.Execute(image)
  
  return newimage

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

def getCentroid(im, label_id):
    """
    Compute the centroid (mean coordinates) of one labelled region
    
    Args:
        im: vtkImage of label map
        label_id: region id
    Returns:
        centroid: np array of the centroid coordinates
    """
    from vtk.util.numpy_support import vtk_to_numpy

    x, y, z = im.GetDimensions()
    pyIm = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    ids = np.array(np.where(pyIm==label_id)).transpose()
    total_num = len(ids)
    
    origin = np.tile(im.GetOrigin(), total_num).reshape(total_num,3)
    spacing = np.tile(im.GetSpacing(), total_num).reshape(total_num,3)
    
    centroid = np.mean(spacing * ids + origin, axis=0)
    return centroid

def locateRegionBoundary(im, label_id1, label_id2):
    """
    Locate the boundary coordinates between two regions with different labels
    
    Args:
        im: vtkImage of the label map
        label_id1: class id of 1st region
        label_id2: class id of 2nd region
        
    Returns
        points: coordinates of the boundary points
    """
    dilateErode = vtk.vtkImageDilateErode3D()
    dilateErode.SetInputData(im)
    dilateErode.SetDilateValue(label_id1)
    dilateErode.SetErodeValue(label_id2)
    
    kernel_size = np.rint(1./np.array(im.GetSpacing())).astype(int)
    print(kernel_size)
    dilateErode.SetKernelSize(*kernel_size)
    dilateErode.Update()
    newIm = dilateErode.GetOutput()
    
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    x, y, z = newIm.GetDimensions()
    pyIm_new = vtk_to_numpy(newIm.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    pyIm = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    ids = np.array(np.where(pyIm_new-pyIm!=0)).transpose()

    total_num = len(ids)
    
    origin = np.tile(newIm.GetOrigin(), total_num).reshape(total_num,3)
    spacing = np.tile(newIm.GetSpacing(), total_num).reshape(total_num,3)
    points = ids * spacing + origin
    return points

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
    
    x, y, z = np.meshgrid(range(Y), range(Z), range(X))
    indices = np.moveaxis(np.vstack((z.flatten(),x.flatten(),y.flatten())),0,1)
    b = np.tile(spacing, total_num).reshape(total_num,3)
    physical = indices * b +np.tile(origin, total_num).reshape(total_num,3)
    vec1 = physical - np.tile(ori, total_num).reshape(total_num,3)
    vec2 = np.tile(nrm, total_num).reshape(total_num,3)
    below = np.sum(vec1*vec2, axis=1)<0
    pyLabel[below] = bg_id
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel))

    return labels

def recolorVTKPixelsByPlaneByRegion(labels, ori, nrm, region_id, bg_id):
    """
    Within each region, for every pixel above a plane in physcal coordinates, change the pixel value to background pixel value

    TO-DO: Mayber combine with the previous function
    Args:
        labels: VTK image
        ori: plane origin
        nrm: plane normal
        region_id: class id of the labelled region
        bg_id: class id of the new color
    Returns:
        labels: editted VTK image
    """

    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    x, y, z = labels.GetDimensions()
    pyLabel = vtk_to_numpy(labels.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    
    indices = np.array(np.where(pyLabel==region_id)).transpose()
    total_num = len(indices)

    spacing = np.tile(labels.GetSpacing(), total_num).reshape(total_num,3)
    origin = np.tile(labels.GetOrigin(), total_num).reshape(total_num,3)
    physical = indices * spacing + origin
   
    vec1 = physical - np.tile(ori, total_num).reshape(total_num,3)
    vec2 = np.tile(nrm, total_num).reshape(total_num,3)
    above = np.sum(vec1*vec2, axis=1)>0

    remove_indices = indices[above]
    for i in remove_indices:
        pyLabel[i[0],i[1],i[2]] = bg_id

    print(pyLabel.shape)
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel.transpose(2,1,0).flatten()))

    return labels

def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given dimenstion

    Args:
        image: vtk Image data
        spacing: image new spacing
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

    #size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    #new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()


def convertVTK2binary(labels):
    """
    This function converts a vtk label to binary label
    
    Args:
        labels: VTK image
    Returns:
        labels: converted VTK image
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    pyLabel = vtk_to_numpy(labels.GetPointData().GetScalars())
    pyLabel[np.where(pyLabel!=0)] = 1
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel))
    return labels
