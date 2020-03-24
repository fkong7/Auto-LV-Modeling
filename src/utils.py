"""
Utility functions for label map editing

@author Fanwei Kong
"""
import numpy as np
import vtk

##########################
## Numpy Utility functions
##########################
def swapLabels(pyImage):
    """
    Swap label ids
    """
    ids = np.unique(pyImage)
    print("UNIQUE IDS: ", ids)
    for i, v in enumerate(ids):
        pyImage[pyImage==v] = i
    return pyImage

def fitPlaneNormal(points_input):
    """
    Fit a plane to a point set
    
    Args:
        points_input: 3d coordinates of a point set
    Returns:
        normal: normal of the fitting plane
    """
    G = points_input.sum(axis=0) / points_input.shape[0]

    u, s, vh = np.linalg.svd(points_input - G)
    normal = vh[2, :]
    return normal

def fitPlaneNormal2(points_input):
    """
    Fit a plane to a point set
    
    Args:
        points_input: 3d coordinates of a point set
    Returns:
        normal: normal of the fitting plane
    """
    from scipy.optimize import minimize
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
    res = minimize(fun, params0)
    a = res.x[0]
    b = res.x[1]
    c = res.x[2]
    point  = np.array([0.0, 0.0, c])
    normal = np.array(_cross([1,0,a], [0,1,b]))
    D = -point.dot(normal)
    return normal

########################
## Label Map functions
#######################def closing(image, ids):
def closing(im, ids):
    import SimpleITK as sitk
    spacing = im.GetSpacing()
    kernel = [int(round(5./spacing[i])) for i in range(3)]
    kernel = [8 if kernel[i]>8 else kernel[i] for i in range(3)]
    ftr = sitk.BinaryMorphologicalClosingImageFilter()
    ftr.SetKernelRadius(kernel)
    ftr.SafeBorderOn()
    for i in ids:
        if i ==0:
            continue
        ftr.SetForegroundValue(int(i))
        im = ftr.Execute(im)
    return im

def resample(image, resolution = (0.5, 0.5, 0.5), dim=3, order=0):
  """
  This function resamples a SimpleITK image to desired resolution

  Args:
    image: SimpleItk image
    resolution: desired grid resolution
    dim: image dimension
  Returns:
    newimage: resampeled SimpleITK image
  """
  import SimpleITK as sitk
  resample = sitk.ResampleImageFilter()
  if order ==0:
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
  else:
    resample.SetInterpolator(sitk.sitkLinear)
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
    import SimpleITK as sitk
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
def decimation(poly, rate):
    """
    Simplifies a VTK PolyData

    Args: 
        poly: vtk PolyData
        rate: target rate reduction
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOn()
    decimate.Update()
    return decimate.GetOutput()

def cleanPolyData(poly, tol):
    """
    Cleans a VTK PolyData

    Args:
        poly: VTK PolyData
        tol: tolerance to merge points
    Returns:
        poly: cleaned VTK PolyData
    """

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(tol)
    clean.PointMergingOn()
    clean.Update()

    poly = clean.GetOutput()
    return poly

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
    
    poly = cleanPolyData(appender, 0.)
    return poly

def smoothVTKPolydata(poly, iteration=10, boundary=False, feature=False):
    """
    This function smooths a vtk polydata

    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool

    Returns:
        smoothed: smoothed vtk polydata
    """

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed

def laplacianSmoothVTKPolydata(poly, iteration=10, boundary=False, feature=False):
    """
    This function smooths a vtk polydata

    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool

    Returns:
        smoothed: smoothed vtk polydata
    """

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed

def windowedSincSmoothVTKPolyData(poly, iteration=15, band=0.1, boundary=False, feature=False):
    """
    This function smooths a vtk polydata, using windowed sinc algorithm

    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool

    Returns:
        smoothed vtk polydata
    """
    ftr = vtk.vtkWindowedSincPolyDataFilter()
    ftr.SetInputData(poly)
    ftr.SetNumberOfIterations(iteration)
    ftr.SetPassBand(band)
    ftr.SetBoundarySmoothing(boundary)
    ftr.SetFeatureEdgeSmoothing(feature)
    ftr.NonManifoldSmoothingOn()
    ftr.NormalizeCoordinatesOn()
    ftr.Update()
    return ftr.GetOutput()
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

    return boolean.GetOutput()

def fillHole(poly, size=10000000.):
    """
    Fill holes in VTK PolyData
    
    Args:
        poly: VTK PolyData to fill
        size: max size of the hole to fill
    Returns:
        poly: filled VTK PolyData
    """
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(poly)
    filler.SetHoleSize(size)
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
    smoother.SetRadiusFactors(np.array(im.GetSpacing())*stdev)
    smoother.Update()
    return smoother.GetOutput()

def labelOpenClose(im, label_id, bg_id, size):
    """
    Open or close 

    Args: 
        im: vtkImage of the label map
        label_id: class id to close
        bg_id: class id of background to open
        size: number of pixels of the erosion
    Return: processed vtk image
    """

    filt = vtk.vtkImageOpenClose3D()
    filt.SetInputData(im)
    filt.SetOpenValue(bg_id)
    filt.SetCloseValue(label_id)
    filt.SetKernelSize(int(size), int(size), int(size))
    filt.Update()
    return filt.GetOutput()

def labelDilateErode(im, label_id, bg_id, size):
    """
    Dilates a label
    
    Args:
        im: vtkImage of the label map
        label_id: class id to erode
        bg_id: class id of backgroud to dilate
        size: num of pixels of the erosion
    Returns
        newIm: vtkImage with dilated tissue structure
    """
    dilateErode = vtk.vtkImageDilateErode3D()
    dilateErode.SetInputData(im)
    dilateErode.SetDilateValue(label_id)
    dilateErode.SetErodeValue(bg_id)
    
    #kernel_size = np.rint(thickness/np.array(im.GetSpacing())).astype(int)
    kernel_size = (np.ones(3) * size).astype(int)
    dilateErode.SetKernelSize(*kernel_size)
    dilateErode.Update()
    newIm = dilateErode.GetOutput()

    return newIm
    

def createTissueThickness(im, label_id, bg_id,thickness,binary=True):
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

    newIm = labelDilateErode(im, label_id, bg_id,thickness)
    
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    pyIm_new = vtk_to_numpy(newIm.GetPointData().GetScalars())
    pyIm = vtk_to_numpy(im.GetPointData().GetScalars())
    pyIm_new[np.where((pyIm_new-pyIm==0)&(pyIm==label_id))]=bg_id
    if binary:
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

def locateRegionBoundaryIDs(im, label_id1, label_id2, size = 1., bg_id = None):
    """
    Locate the boundary coordinates between two regions with different labels
    
    Args:
        im: vtkImage of the label map
        label_id1: class id of 1st region
        label_id2: class id of 2nd region
        bg_id: class id of background
    Returns
        ids: ids of the boundary points
    """
    new_Im = vtk.vtkImageData()
    new_Im.DeepCopy(im)
    if bg_id is not None:
        dilateErode = vtk.vtkImageDilateErode3D()
        dilateErode.SetInputData(new_Im)
        dilateErode.SetDilateValue(label_id1)
        dilateErode.SetErodeValue(bg_id)
        dilateErode.SetKernelSize(3, 3, 3)
        dilateErode.Update()
        new_Im = dilateErode.GetOutput()
    
    dilateErode = vtk.vtkImageDilateErode3D()
    dilateErode.SetInputData(new_Im)
    dilateErode.SetDilateValue(label_id1)
    dilateErode.SetErodeValue(label_id2)
    
    #kernel_size = np.rint(size/np.array(im.GetSpacing())).astype(int)
    kernel_size = (np.ones(3)*size).astype(int)
    dilateErode.SetKernelSize(*kernel_size)
    dilateErode.Update()
    newIm = dilateErode.GetOutput()

    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    x, y, z = newIm.GetDimensions()
    pyIm_new = vtk_to_numpy(newIm.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    pyIm = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    ids = np.array(np.where(pyIm_new-pyIm!=0)).transpose()
    return ids


def locateRegionBoundary(im, label_id1, label_id2, size=1.):
    """
    Locate the boundary coordinates between two regions with different labels
    
    Args:
        im: vtkImage of the label map
        label_id1: class id of 1st region
        label_id2: class id of 2nd region
        
    Returns
        points: coordinates of the boundary points
    """
    ids = locateRegionBoundaryIDs(im, label_id1, label_id2, size)
    
    total_num = len(ids)
    
    origin = np.tile(im.GetOrigin(), total_num).reshape(total_num,3)
    spacing = np.tile(im.GetSpacing(), total_num).reshape(total_num,3)
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

def recolorVTKPixelsByPlane(labels, ori, nrm, bg_id, label_id=None):
    """
    For every pixel above a plane in physcal coordinates, change the pixel value to background pixel value

    Args:
        labels: VTK image
        ori: plane origin
        nrm: plane normal
        bg_id: id to erode the label to
        label_id: id to avoid erasing
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
    above = np.sum(vec1*vec2, axis=1)>0
    if label_id is not None:
        above = np.logical_and(above, pyLabel!=label_id)
    pyLabel[above] = bg_id
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel))

    return labels

def recolorVTKPixelsByIds(labels, ids, bg_id):
    """
    Change the pixel values of the specified ids to background pixel value

    Args:
        labels: VTK image
        ids: ids to change
        bg_id: class id to change to
    Returns: editted VTK image
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    x, y, z = labels.GetDimensions()
    pyLabel = vtk_to_numpy(labels.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
  
    for i in ids:
        pyLabel[i[0], i[1], i[2]] = bg_id
    labels.GetPointData().SetScalars(numpy_to_vtk(pyLabel.transpose(2,1,0).flatten()))

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


def extractLargestConnectedRegion(vtk_im, label_id):
    """
    Extrac the largest connected region of a vtk image

    Args:
        vtk_im: vtk image
        label_id: id of the label
    Return:
        new_im: processed vtk image
    """

    fltr = vtk.vtkImageConnectivityFilter()
    fltr.SetScalarRange(label_id, label_id)
    fltr.SetExtractionModeToLargestRegion()
    fltr.SetInputData(vtk_im)
    fltr.Update()
    new_im = fltr.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    py_im = vtk_to_numpy(vtk_im.GetPointData().GetScalars())
    py_mask = vtk_to_numpy(new_im.GetPointData().GetScalars())
    mask = np.logical_and(py_im==label_id, py_mask==0)
    py_im[mask] = 0
    vtk_im.GetPointData().SetScalars(numpy_to_vtk(py_im))
    return vtk_im

def cutPolyDataWithAnother(poly1, poly2, inside=False):
    """
    Cuts the first VTK PolyData with another
    
    Args:
        poly1: 1st VTK PolyData
        poly2: 2nd VTK PolyData
        inside: whether inside or outside gets kept (bool)
    Returns:
        poly: cut VTK PolyData
    """
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(poly2)

    #clipper = vtk.vtkClipPolyData()
    clipper = vtk.vtkExtractPolyDataGeometry()
    #clipper.SetClipFunction(implicit)
    clipper.SetImplicitFunction(implicit)
    clipper.SetInputData(poly1)
    #clipper.SetInsideOut(inside)
    clipper.SetExtractInside(inside)
    clipper.SetExtractBoundaryCells(True)
    clipper.Update()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(clipper.GetOutput())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def boundaryEdges(mesh):
    """
    Finds boundary edges of a VTK PolyData

    Args:
        mesh: VTK PolyData
    
    Returns:
        edges: Extracted boundary edges as VTK PolyData
    """

    extractor = vtk.vtkFeatureEdges()
    extractor.SetInputData(mesh)
    extractor.BoundaryEdgesOn()
    extractor.FeatureEdgesOff()
    extractor.NonManifoldEdgesOff()
    extractor.ManifoldEdgesOff()
    extractor.SetFeatureAngle(150)
    extractor.Update()
    return extractor.GetOutput()

def findPointCorrespondence(mesh,points):
    """
    Find the point IDs of the points on a VTK PolyData

    Args:
        mesh: the PolyData to find IDs on
        points: vtk Points
    
    Returns
        IdList: list containing the IDs
    TO-DO: optimization move vtkKdTreePointLocator out of the loop, why
    is it inside now?
    """
    IdList = [None]*points.GetNumberOfPoints()
    for i in range(len(IdList)):
        newPt = points.GetPoint(i)
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        IdList[i] = locator.FindClosestPoint(newPt)

    return IdList


def separateDisconnectedPolyData(poly):
    """
    Separate disconnected PolyData into separate PolyData objects

    Args:
        poly: VTK PolyData
    Returns:
        components: list of VTK PolyData objects
    """
    cc_filter = vtk.vtkPolyDataConnectivityFilter()
    cc_filter.SetInputData(poly)
    cc_filter.SetExtractionModeToSpecifiedRegions()
    components = list()
    idx = 0
    while True:
        cc_filter.AddSpecifiedRegion(idx)
        cc_filter.Update()
        
        component = vtk.vtkPolyData()
        component.DeepCopy(cc_filter.GetOutput())
        component = cleanPolyData(component, 0.)
        # Make sure we got something
        if component.GetNumberOfCells() <= 0:
            break
        components.append(component)
        cc_filter.DeleteSpecifiedRegion(idx)
        idx += 1
    return components

def getPointIdsOnBoundaries(poly):
    """
    Get the point IDs on connected boundaries
    
    Args:
        poly: VTK PolyData
    Returns:
        id_lists: a list of Python lists, each containing the point IDs of one connected boundary (e.g., mitral opening)
        pt_lists: a list of vtk Points, each containing the points of one connected boundary
    """
    edges = boundaryEdges(poly)
    components = separateDisconnectedPolyData(edges)
    id_lists = [None]*len(components)
    #pt_lists = [None]*len(components)
    for i in range(len(id_lists)):
        id_lists[i] = findPointCorrespondence(poly,components[i].GetPoints())
        #pt_lists[i] = components[i].GetPoints()
        print('Found %d points for boundary %d\n' % (len(id_lists[i]),i))
    return id_lists,components

def changePolyDataPointsCoordinates(poly, pt_ids, pt_coords):
    """
    For points with ids of a VTK PolyData, change their coordinates

    Args:
        poly: vtkPolyData
        pt_ids: id lists of the points to change, python list
        pt_coords: corresponding point coordinates of the points to change, numpy array
    Returns:
        poly: vtkPolyData after changing the points coordinates
    """
    from vtk.util.numpy_support import vtk_to_numpy
    if type(pt_coords)==vtk.vtkPoints():
        pt_coords = vtk_to_numpy(pt_coords.GetData())
    if len(pt_ids)!=pt_coords.shape[0]:
        raise ValueError('Number of points do not match')
        return
    for i, idx in enumerate(pt_ids):
        poly.GetPoints().SetPoint(idx, pt_coords[i,:])

    return poly

def projectPointsToFitPlane(points):
    """
    Find the best fit plane of VTK points, project the points to the plane

    NOTE (REMOVED, NOT GENERALIZED): The origin of the plane is the point centroid offset by the largest positive distance of the points to the fit plane

    Args:
        points: vtkPoints
        #ref: a reference point above the plane (helps to determine the direction of normal)
    Returns:
        pyPts: projected points in Python
    """

    from vtk.util.numpy_support import vtk_to_numpy
    # find normal and origin
    if type(points)==np.ndarray:
        pyPts = points
    else:
        pyPts = vtk_to_numpy(points.GetData())
    nrm = fitPlaneNormal(pyPts)
    nrm /= np.linalg.norm(nrm)
    ori = np.mean(pyPts, axis=0)

    #if np.dot(nrm, ref-ori)<0:
    #    nrm = -1*nrm


    num = pyPts.shape[0]
    #distance = np.sum((pyPts-np.repeat(ori[np.newaxis,:],num,axis=0))
    #            * np.repeat(nrm[np.newaxis,:],num,axis=0),axis=1)
    #ori += np.max(distance)*nrm

    plane = vtk.vtkPlane()
    plane.SetOrigin(*ori)
    plane.SetNormal(*nrm)

    proj_Pts = np.zeros(pyPts.shape)

    for i in range(pyPts.shape[0]):
        plane.ProjectPoint(pyPts[i,:],proj_Pts[i,:])
    
    return proj_Pts

def smoothVTKPolyline(polyline, smooth_iter):
    """
    smooth the points on rings, works for mitral or aorta opening like geometry

    Args:
        polyline: vtk polydata of a polyline
        smooth_iter: smoothing iteration
    Returns:
        polyline: smoothed vtk polyline
    """
    for ITER in range(smooth_iter):
        for i in range(polyline.GetNumberOfPoints()):
            neighbors = vtk.vtkIdList()
            cell_ids = vtk.vtkIdList()
            polyline.GetPointCells(i, cell_ids)
            for j in range(cell_ids.GetNumberOfIds()):
                pt_ids = vtk.vtkIdList()
                polyline.GetCellPoints(cell_ids.GetId(j), pt_ids)
                for k in range(pt_ids.GetNumberOfIds()):
                    idx = pt_ids.GetId(k)
                    if idx != i:
                        neighbors.InsertNextId(idx)
            assert neighbors.GetNumberOfIds() == 2 , ("Found incorrect num of neighbors:", neighbors.GetNumberOfIds())
            pt1 = polyline.GetPoints().GetPoint(neighbors.GetId(0))
            pt2 = polyline.GetPoints().GetPoint(neighbors.GetId(1))
            # average of the two neighbors
            pt = 0.5*np.array(pt1)+0.5*np.array(pt2)
            polyline.GetPoints().SetPoint(i, pt)
    return polyline

class pointLocator:
    # Class to find closest points
    def __init__(self,pts):
        ds = vtk.vtkPolyData()
        ds.SetPoints(pts)
        self.locator = vtk.vtkKdTreePointLocator()
        self.locator.SetDataSet(ds)
        self.locator.BuildLocator()
    def findNClosestPoints(self,pt, N):
        ids = vtk.vtkIdList()
        self.locator.FindClosestNPoints(N, pt, ids)
        return ids

def projectOpeningToFitPlane(poly, boundary_ids, points, MESH_SIZE):
    """
    This function projects the opening geometry to a best fit plane defined by the points on opennings. Differenet from the previous function, not only the points on openings were moved but the neighbouring nodes to maintain mesh connectivity.
    Args:
        poly: VTK PolyData
        boundary_ids: boundary ids
        points: bounary pts, vtk points or numpy
        MESH_SIZE: mesh edge size, used to find the number of times to find connected points and move them
    Returns:
        poly: VTK PolyData of the modified geometry
    """
    
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    #if numpy convert to vtk
    if type(points)==np.ndarray:
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(points))
    else:
        pts = points

        
    def _moveConnectedPoints(ids, pts, proj_pts, factor):
        locator = pointLocator(pts)
        displacement = proj_pts - vtk_to_numpy(pts.GetData())
        for i in range(pts.GetNumberOfPoints()):
            cell_ids = vtk.vtkIdList()
            poly.GetPointCells(ids[i], cell_ids)
            connected_pt_ids = vtk.vtkIdList()
            for j in range(cell_ids.GetNumberOfIds()):
                poly.GetCellPoints(cell_ids.GetId(j), connected_pt_ids)
                for k in range(connected_pt_ids.GetNumberOfIds()):
                    pt_id = connected_pt_ids.GetId(k)
                    if pt_id not in ids:
                        ids.append(pt_id)
                        # find direction, mean of displacement bewteen this pt and two closest points
                        pt = poly.GetPoints().GetPoint(pt_id)
                        pts.InsertNextPoint(pt)
                        close_pts = locator.findNClosestPoints(pt, 2)
                        pt += (displacement[close_pts.GetId(0),:]+displacement[close_pts.GetId(1)]) * factor/2
                        proj_pts = np.vstack((proj_pts, pt))
        return ids, pts, proj_pts
    
    #make a copy of the pt ids
    ids = boundary_ids.copy()
    proj_pts = projectPointsToFitPlane(pts)
    dist = np.max(np.linalg.norm(proj_pts - vtk_to_numpy(pts.GetData()), axis=1))
    ITER = np.ceil(dist/MESH_SIZE)*3
    print("ITER: ", ITER)
    for factor in np.linspace(0.8, 0., ITER, endpoint=False):
        ids, pts,  proj_pts = _moveConnectedPoints(ids, pts, proj_pts, factor)
    poly = changePolyDataPointsCoordinates(poly, ids, proj_pts)
    return poly 

def getPolyDataPointCoordinatesFromIDs(poly, pt_ids):
    """
    Return the coordinates of points of the PolyData with ids

    Args:
        poly: vtkPolyData
        pt_ids: id lists of the points
    Returns:
        pts: numpy array containing the point coordinates
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    poly_points = vtk_to_numpy(poly.GetPoints().GetData())

    pts = poly_points[pt_ids,:]
    return pts


def deleteCellsFromPolyData(poly,id_list):
    """
    Removes cells from VTK PolyData with their id numbers

    Args:
        poly: VTK PolyData
        id_list: python list of cell id numbers
    Returns:
        poly: VTK PolyData with cells removed
    """
    poly.BuildLinks()
    for idx in id_list:
        poly.DeleteCell(idx)
    poly.RemoveDeletedCells()
    poly.DeleteLinks()
    return poly

def removeFreeCells(poly, pt_ids):
    """
    Removes cells that only have one edge attached to the mesh connected to a point cell
    For each point (identified by id numebr) on mesh, find the number of connected triangles
    If only one triangle is found, remove this trangle

    Args:
        poly: VTK PolyData
        pt_ids: python list of point ids
    Returns:
        poly: VTK PolyData with removed cells
        pt_ids: python list of point ids after removing points on the removed cells
    """
    cell_list = list()
    for idx in pt_ids:
        id_list = vtk.vtkIdList()
        poly.GetPointCells(idx, id_list)
        if id_list.GetNumberOfIds()==1:
            cell_list.append(id_list.GetId(0))
            pt_ids.remove(idx)
    poly = deleteCellsFromPolyData(poly, cell_list)
    return poly, pt_ids

def cutSurfaceWithPolygon(poly, boundary):
    """
    UNDER-DEVELOPMENT... ONLY WORKS FOR 2D DELAUNAY RESULTS
    Cuts a surface with a polygon and removes cells outside the polygon
    Only tested for 2D surface and 2D polygon

    Args:
        poly: flat surface, VTK PolyData
        boundary: boundary edge to trim the polydaya, VTK PolyData
    Returns:
        poly: trimed VTK PolyData
    """
    #from vtk.util.numpy_support import vtk_to_numpy
    delete_list = list()
    #bound_pts = vtk_to_numpy(boundary.GetPoints().GetData())
    bound_ids = findPointCorrespondence(poly,boundary.GetPoints())
    #poly_pts = vtk_to_numpy(poly.GetPoints().GetData())
    #ctr = np.mean(bound_pts, axis=0)
    for idx in range(poly.GetNumberOfCells()):
        pt_ids = vtk.vtkIdList()
        poly.GetCellPoints(idx, pt_ids)

        num = 0
        #py_id = list()
        for i in range(pt_ids.GetNumberOfIds()):
            if pt_ids.GetId(i) in bound_ids:
                num+=1
                #py_id.append(pt_ids.GetId(i))
        if num == 3:
            delete_list.append(idx)
    poly = deleteCellsFromPolyData(poly, delete_list)

    return poly

def deleteBadQualityCells(poly, tol):
    """
    Deletes cells with bad mesh quality (minimum angle)

    Args:
        poly: VTK PolyData to process
        tol: minimum angles to threshold to delete
    Returns:
        poly: VTK PolyData with deleted cells
    """
    qfilter = vtk.vtkMeshQuality()
    qfilter.SetInputData(poly)
    qfilter.SetTriangleQualityMeasureToMinAngle()
    qfilter.Update()
    angle = qfilter.GetOutput().GetCellData().GetArray("Quality")
    from vtk.util.numpy_support import vtk_to_numpy
    pyangle = vtk_to_numpy(angle)
    ids = [i for i in range(len(pyangle)) if pyangle[i]<tol]
    print("Bad element, deleting...", ids)
    poly = deleteCellsFromPolyData(poly,ids)
    return poly

def appendPolyData(poly1, poly2):
    """
    Combine two VTK PolyData objects together
    Args:
        poly1: first PolyData
        poly2: second PolyData
    Return:
        poly: combined PolyData
    """
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(poly1)
    appendFilter.AddInputData(poly2)
    appendFilter.Update()
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputData(appendFilter.GetOutput())
    cleanFilter.SetTolerance(0.0)
    cleanFilter.PointMergingOn()
    cleanFilter.Update()
    poly = cleanFilter.GetOutput()
    return poly

def tagPolyData(poly, tag):
    """
    Tag polydata with a tag id

    Args:
        poly: VTK PolyData
        tag: tag id (int)
    Returns:
        poly: tagged PolyData
    """
    tags = vtk.vtkIntArray()
    tags.SetNumberOfComponents(1)
    tags.SetName('ModelFaceID')
    tags.SetNumberOfValues(poly.GetNumberOfPolys())
    for i in range(poly.GetNumberOfPolys()):
        tags.SetValue(i, int(tag))
    poly.GetCellData().SetScalars(tags)
    return poly

def fixPolydataNormals(poly):
    """
    Adjust the normals of PolyData

    Args:
        poly: vtk PolyData
    Returns:
        poly: adjusted vtk PolyData
    """
    normAdj = vtk.vtkPolyDataNormals()
    normAdj.SetInputData(poly)
    normAdj.SplittingOff()
    normAdj.ConsistencyOn()
    normAdj.FlipNormalsOn()
    normAdj.Update()
    poly = normAdj.GetOutput()
    return poly
def orientedPointsetFromBoundary(boundary):
    """
    Create list of oriented ids on a closed boundary curve (polyline)

    Args: 
        boundary: VTK PolyData
    Returns:
        id_list: list of ordered ids
        pt_list: vtk points with ordered points
    """
    bound_pts = boundary.GetPoints()
    pt_list = vtk.vtkPoints()
    pt_list.SetNumberOfPoints(bound_pts.GetNumberOfPoints())
    id_list = [None]*bound_pts.GetNumberOfPoints()
    pt_list.SetPoint(0, bound_pts.GetPoint(0))
    id_list[0] = 0

    cells = vtk.vtkIdList()
    boundary.GetPointCells(id_list[0], cells)
    pts = vtk.vtkIdList()
    boundary.GetCellPoints(cells.GetId(0), pts)
    for i in range(pts.GetNumberOfIds()):
        if pts.GetId(i) not in id_list:
            id_list[1] = pts.GetId(i)
            pt_list.SetPoint(1, bound_pts.GetPoint(i))

    for i in range(2, len(id_list)):
        cells = vtk.vtkIdList()
        boundary.GetPointCells(id_list[i-1], cells)
        for j in range(cells.GetNumberOfIds()):
            pts = vtk.vtkIdList()
            boundary.GetCellPoints(cells.GetId(j), pts)
            for k in range(pts.GetNumberOfIds()):
                if pts.GetId(k) not in id_list:
                    id_list[i] = pts.GetId(k)
                    pt_list.SetPoint(i, bound_pts.GetPoint(k))
    return id_list, pt_list

def capPolyDataOpenings(poly,  size):
    """
    Cap the PolyData openings  with acceptable mesh quality

    Args:
        poly: VTK PolyData to cap
        size: edge size of the cap mesh
    Returns:
        poly: capped VTK PolyData
    """
    # TRY NOT USE TO USE THE POINT IDS, START FROM FEATURE EDGE DIRECTLY SINCE IT REQUIRES THE BOUNDARY POLYDATA
    #import matplotlib.pyplot as plt
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    import label_io
    import os
    def _plotPoints(points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        plt.show()
    def _addNodesToCap(vtkPts, size):
        """
        Add uniformed points to cap
        """
        points = vtk_to_numpy(vtkPts.GetData())
        num = points.shape[0]
        #_plotPoints(points)
        ctr = np.mean(points, axis=0)
        length = np.mean(np.linalg.norm(points-ctr, axis = 1))
        r = np.linspace(0.5*size/length, (length-size*0.8)/length,np.floor(length/size))
        addedPts = vtk.vtkPoints()
        for rf in r:
            newPts = vtk.vtkPoints()
            newPts.SetData(numpy_to_vtk((points-ctr)*rf+ctr))
            addedPts.InsertPoints(addedPts.GetNumberOfPoints()
                                    ,newPts.GetNumberOfPoints()
                                    ,0,newPts)
        ptsPly = vtk.vtkPolyData()
        ptsPly.SetPoints(addedPts)
        vertexFilter = vtk.vtkVertexGlyphFilter()
        vertexFilter.SetInputData(ptsPly)
        vertexFilter.Update()
        ptsPly = vertexFilter.GetOutput()
        cleanedPts = cleanPolyData(ptsPly, size*0.01)

        vtkPts.InsertPoints(vtkPts.GetNumberOfPoints()
                            ,cleanedPts.GetNumberOfPoints()
                            ,0
                            ,cleanedPts.GetPoints())
            
        #_plotPoints(vtk_to_numpy(vtkPts.GetData()))
        return vtkPts

    def _delaunay2D(vtkPts, boundary):
        """
        Delaunay 2D on input points
        """
        vtkPtsPly = vtk.vtkPolyData()
        vtkPtsPly.SetPoints(vtkPts)
        
        ids, pt_list = orientedPointsetFromBoundary(boundary)   

        polygon = vtk.vtkCellArray()
        polygon.InsertNextCell(len(ids))
        for i in ids:
            polygon.InsertCellPoint(i)
        vtkPtsPly.SetPolys(polygon)
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(vtkPtsPly)
        delaunay.SetSourceData(vtkPtsPly)
        delaunay.SetTolerance(0.002)
        delaunay.Update()
        return delaunay.GetOutput()
    #tag polydata 
    tag_id = 1
    poly = tagPolyData(poly, tag_id)

    edges = boundaryEdges(poly)
    components = separateDisconnectedPolyData(edges)
    id_lists = [None]*len(components)
    pt_lists = [None]*len(components)
    for i in range(len(id_lists)):
        id_lists[i] = findPointCorrespondence(poly,components[i].GetPoints())
        pt_lists[i] = vtk.vtkPoints()
        pt_lists[i].DeepCopy(components[i].GetPoints())
        print('Found %d points for boundary %d\n' % (len(id_lists[i]),i))
   
    cap_pts_list = list()
    for boundary, ids, pts in zip(components, id_lists, pt_lists):
        cap_pts = _addNodesToCap(pts, size)
        cap_pts_list.append(cap_pts)
        cap = _delaunay2D(cap_pts, boundary)
        #cap = cutSurfaceWithPolygon(cap, boundary)
        #tag the caps
        tag_id +=1
        cap = tagPolyData(cap, tag_id)
        poly = appendPolyData(poly, cap)
    
    #cap_pts_ids = list()
    #for cap_pts in cap_pts_list:
    #    cap_pts_ids.append(findPointCorrespondence(poly,cap_pts))
    poly = fixPolydataNormals(poly)
    return poly

def getPolydataVolume(poly):
    """
    Compute volume of a enclosed vtk polydata

    Args:
        poly: vtk PolyData
    Returns:
        volume: PolyData volume
    """

    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    volume = mass.GetVolume()
    return volume

def thresholdPolyData(poly, attr, threshold):
    """
    Get the polydata after thresholding based on the input attribute

    Args:
        poly: vtk PolyData to apply threshold
        atrr: attribute of the cell array
        threshold: (min, max) 
    Returns:

        output: resulted vtk PolyData
    """
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    surface_thresh.SetInputArrayToProcess(0,0,0,1,attr)
    surface_thresh.ThresholdBetween(*threshold)
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()

