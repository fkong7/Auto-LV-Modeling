"""
Utility functions for label map editing

@author Fanwei Kong
"""
import numpy as np
import vtk

def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

##########################
## Numpy Utility functions
##########################
def swap_labels(pyImage):
    """
    Swap label ids
    """
    ids = np.unique(pyImage)
    for i, v in enumerate(ids):
        pyImage[pyImage==v] = i
    return pyImage

def fit_plane_normal(points_input):
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

########################
## Label Map functions
#######################def closing(image, ids):
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

def convert_to_binary(labels):
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

def normalize_label_map(labels, values=[], keep=[]):
    """
    Normalize the intensity value of segmentation to a specified range
    Args:
        labels: SimpleITK segmentation
    """
    import SimpleITK as sitk
    #labels = sitk.Cast(labels,  sitk.sitkFloat32)
    py_label = sitk.GetArrayFromImage(labels)

    ids = np.unique(py_label)
    #values = np.linspace(rng[0], rng[1], len(keep), endpoint=True) #if keep is empty, convert to binary
    for index, i in enumerate(ids):
        if i in keep:
            py_label[py_label==i] = values[keep.index(i)]
        else:
            py_label[py_label==i] = 0.
    labels_new = sitk.GetImageFromArray(py_label)
    labels_new.SetOrigin(labels.GetOrigin())
    labels_new.SetDirection(labels.GetDirection())
    labels_new.SetSpacing(labels.GetSpacing())
    return labels_new

def erase_boundary(labels, pixels, bg_id):
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

def remove_class(labels, class_id, bg_id):
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

################################
## VTK PolyData functions
###############################
def clean_polydata(poly, tol):
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

def smooth_vtk_polydata(poly, iteration=10, boundary=False, feature=False):
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

def windowed_sinc_smooth_vtk_polydata(poly, iteration=15, band=0.1, boundary=False, feature=False):
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
def boolean_vtk_polydata(poly1, poly2, keyword):
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

def fill_hole(poly, size=10000000.):
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

def label_open_close(im, label_id, bg_id, size):
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

def label_dilate_erode(im, label_id, bg_id, size):
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
    

def constrained_local_smoothing(poly, ctr, radius, iteration, factor):
    for index in range(poly.GetNumberOfPoints()):
        pt = np.array(poly.GetPoints().GetPoint(index))
        if np.linalg.norm(pt-ctr)<radius:
            for it in range(iteration):
                ptList=vtk.vtkIdList()
                tempList=vtk.vtkIdList()
                tempPtList=vtk.vtkIdList()
                tempCoord=np.zeros((0,3))
                poly.GetPointCells(index,tempList)
                for i in range(tempList.GetNumberOfIds()):
                    poly.GetCellPoints(tempList.GetId(i),tempPtList)
                    for j in range(3):
                        if tempPtList.GetId(j)!=index:
                            ptList.InsertUniqueId(tempPtList.GetId(j))
                for i in range(ptList.GetNumberOfIds()):
                    tempCoord = np.vstack((tempCoord, np.array(poly.GetPoints().GetPoint(ptList.GetId(i)))))
                disp_vec = np.sum(tempCoord, axis=0)/tempCoord.shape[0] - pt
                poly.GetPoints().SetPoint(index, pt+factor*disp_vec)
    return poly

def get_centroid(im, label_id):
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

def locate_region_boundary_ids(im, label_id1, label_id2, size = 1., bg_id = None):
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
    ids = locate_region_boundary_ids(im, label_id1, label_id2, size)
    
    total_num = len(ids)
    
    origin = np.tile(im.GetOrigin(), total_num).reshape(total_num,3)
    spacing = np.tile(im.GetSpacing(), total_num).reshape(total_num,3)
    points = ids * spacing + origin
    return points


def recolor_vtk_pixels_by_plane(labels, ori, nrm, bg_id, label_id=None):
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

def recolor_vtk_pixels_by_ids(labels, ids, bg_id):
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


def recolor_vtk_image_by_polydata(poly, vtk_image, new_id):
    """
    Change the id value of the pixels within the Polydata

    Args:
        poly: VTK PolyData
        vtk_image: VTK ImageData
        new_id: int, new id
    Return:
        new_image: modified vtk image
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    poly_im = convert_polydata_to_image_data(poly, vtk_image)
    poly_im_py = vtk_to_numpy(poly_im.GetPointData().GetScalars())
    vtk_im_py = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    vtk_im_py[poly_im_py>0] = new_id
    vtk_image.GetPointData().SetScalars(numpy_to_vtk(vtk_im_py))
    return vtk_image

def vtk_image_resample(image, spacing, opt):
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


def convert_vtk_im_to_binary(labels):
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


def extract_largest_connected_region(vtk_im, label_id):
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

def cut_polydata_with_another(poly1, poly2, plane_info):
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
    
    signedDistances = vtk.vtkFloatArray()
    signedDistances.SetNumberOfComponents(1)
    signedDistances.SetName("SignedDistances")
    
    # Evaluate the signed distance function at all of the grid points
    points = poly1.GetPoints()

    ctr, nrm = plane_info
    for pointId in range(points.GetNumberOfPoints()):
        p = points.GetPoint(pointId)
        signedDistance = implicit.EvaluateFunction(p)
        plane_sign = np.sum(np.array(p-ctr)*nrm)
        signedDistance = np.abs(signedDistance) if plane_sign<0 else signedDistance
        signedDistances.InsertNextValue(signedDistance)
    poly1.GetPointData().SetScalars(signedDistances)

    ##clipper = vtk.vtkClipPolyData()
    #clipper = vtk.vtkExtractPolyDataGeometry()
    ##clipper.SetClipFunction(implicit)
    #clipper.SetImplicitFunction(implicit)
    #clipper.SetInputData(poly1)
    ##clipper.SetInsideOut(inside)
    #clipper.SetExtractInside(inside)
    #clipper.SetExtractBoundaryCells(True)
    #clipper.Update()
    p2c= vtk.vtkPointDataToCellData()
    p2c.SetInputData(poly1)
    p2c.Update()
    poly1=p2c.GetOutput()

    clipper = threshold_polydata(poly1, 'SignedDistances', (0., np.inf))   
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    #connectivity.SetInputData(clipper.GetOutput())
    connectivity.SetInputData(clipper)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def find_boundary_edges(mesh):
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

def find_point_correspondence(mesh,points):
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


def separate_disconnected_polydata(poly):
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
        component = clean_polydata(component, 0.)
        # Make sure we got something
        if component.GetNumberOfCells() <= 0:
            break
        components.append(component)
        cc_filter.DeleteSpecifiedRegion(idx)
        idx += 1
    return components

def get_point_ids_on_boundaries(poly):
    """
    Get the point IDs on connected boundaries
    
    Args:
        poly: VTK PolyData
    Returns:
        id_lists: a list of Python lists, each containing the point IDs of one connected boundary (e.g., mitral opening)
        pt_lists: a list of vtk Points, each containing the points of one connected boundary
    """
    edges = find_boundary_edges(poly)
    components = separate_disconnected_polydata(edges)
    id_lists = [None]*len(components)
    #pt_lists = [None]*len(components)
    for i in range(len(id_lists)):
        id_lists[i] = find_point_correspondence(poly,components[i].GetPoints())
        #pt_lists[i] = components[i].GetPoints()
        print('Found %d points for cap %d\n' % (len(id_lists[i]),i))
    return id_lists,components

def change_polydata_points_coordinates(poly, pt_ids, pt_coords):
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

def project_points_to_fit_plane(points, ref):
    """
    Find the best fit plane of VTK points, project the points to the plane

    NOTE (REMOVED, NOT GENERALIZED): The origin of the plane is the point centroid offset by the largest positive distance of the points to the fit plane

    Args:
        points: vtkPoints
        #ref: a reference point above the plane (helps to determine the direction of normal)An estimated normal of the plane
    Returns:
        pyPts: projected points in Python
    """

    from vtk.util.numpy_support import vtk_to_numpy
    # find normal and origin
    if type(points)==np.ndarray:
        pyPts = points
    else:
        pyPts = vtk_to_numpy(points.GetData())
    nrm = fit_plane_normal(pyPts)
    nrm /= np.linalg.norm(nrm)
    ori = np.mean(pyPts, axis=0)

    if np.dot(nrm, ref)<0:
        nrm = -1*nrm


    num = pyPts.shape[0]
    distance = np.sum((pyPts-np.repeat(ori[np.newaxis,:],num,axis=0))
                * np.repeat(nrm[np.newaxis,:],num,axis=0),axis=1)
    ori += np.max(distance)*nrm

    plane = vtk.vtkPlane()
    plane.SetOrigin(*ori)
    plane.SetNormal(*nrm)

    proj_Pts = np.zeros(pyPts.shape)

    for i in range(pyPts.shape[0]):
        plane.ProjectPoint(pyPts[i,:],proj_Pts[i,:])
    
    return proj_Pts

def smooth_vtk_polyline(polyline, smooth_iter):
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

class PointLocator:
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

def get_surface_normals(poly):
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(poly)
    norms.ComputePointNormalsOn()
    norms.ComputeCellNormalsOff()
    norms.ConsistencyOn()
    norms.SplittingOff()
    #norms.FlipNormalsOn()
    #norms.AutoOrientNormalsOn()
    norms.Update()
    poly = norms.GetOutput()
    return poly

def project_opening_to_fit_plane(poly, boundary_ids, points, MESH_SIZE):
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

        
    def _move_connected_points(ids, pts, proj_pts, factor):
        locator = PointLocator(pts)
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
   
    # cap normals
    poly = get_surface_normals(poly)
    normals_mean = np.mean(vtk_to_numpy(poly.GetPointData().GetArray('Normals'))[boundary_ids,:], axis=0)
    #make a copy of the pt ids
    ids = boundary_ids.copy()
    proj_pts = project_points_to_fit_plane(pts, normals_mean)
    dist = np.max(np.linalg.norm(proj_pts - vtk_to_numpy(pts.GetData()), axis=1))
    ITER = int(np.ceil(dist/MESH_SIZE)*3)
    for factor in np.linspace(0.8, 0., ITER, endpoint=False):
        ids, pts,  proj_pts = _move_connected_points(ids, pts, proj_pts, factor)
    poly = change_polydata_points_coordinates(poly, ids, proj_pts)
    return poly 

def get_polydata_point_coordinates_from_ids(poly, pt_ids):
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


def delete_cells_from_polydata(poly,id_list):
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

def remove_free_cells(poly, pt_ids):
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
    poly = delete_cells_from_polydata(poly, cell_list)
    return poly, pt_ids

def extract_polydata_faces(poly, angle, expect_num=None):
    """
    Extract faces of a VTK PolyData based on feature angle
    
    Args:
        poly: VTK PolyData
        angle: feature angle
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    copy = vtk.vtkPolyData()
    copy.DeepCopy(poly)

    normals = vtk.vtkPolyDataNormals()
    normals.SetFeatureAngle(angle)
    normals.SetInputData(poly)
    normals.Update()

    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputConnection(normals.GetOutputPort())
    connectivity.ColorRegionsOn()
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    num_surf = connectivity.GetNumberOfExtractedRegions()
    extracted_regions = connectivity.GetOutput()

    p2c= vtk.vtkPointDataToCellData()
    p2c.SetInputData(extracted_regions)
    p2c.Update()
    extracted_regions = p2c.GetOutput()

    counts  = []
    face_list = []
    for i in range(num_surf):
        face = threshold_polydata(extracted_regions, 'RegionId', (i,i))
        face_list.append(face)
        counts.append(face.GetNumberOfCells())
    orders = np.argsort(counts)[::-1].astype(int)
    if expect_num is None:
        expect_num = num_surf
    saved_id = list(orders[:expect_num])
    face_list = [face_list[i] for i in orders[:expect_num]]
    corr_list = []
    for i in range(expect_num):
        corr_list.append(find_point_correspondence(copy, face_list[i]))
    tags = vtk.vtkIntArray()
    tags.SetNumberOfComponents(1)
    tags.SetName('ModelFaceID')
    tags.SetNumberOfValues(copy.GetNumberOfCells())
    for i in range(copy.GetNumberOfCells()):
        pt_ids = vtk.vtkIdList()
        copy.GetCellPoints(i, pt_ids)
        pt_ids_py = []
        for j in range(pt_ids.GetNumberOfIds()):
            pt_ids_py.append(pt_ids.GetId(j))
        tags.SetValue(i, 1)
        for k in range(expect_num, 1, -1):
            if set(pt_ids_py).issubset(set(corr_list[k-1])):
                tags.SetValue(i, k)
        
    copy.GetCellData().SetScalars(tags)
        
    return copy 

def append_polydata(poly1, poly2):
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

def tag_polydata(poly, tag):
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

def fix_polydata_normals(poly):
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
def subdivision_with_caps(poly,mode,num,cap_id=[2,3], wall_id=1, clean=True):
    """
    Subvidie the polydata while perserving the sharp edges between the cap and the wall

    Args:
        poly: VTK PolyData
        mode: str, loop, linear, butterfly
        cap_id: id list of caps
        wall_id: id of wall --TO-DO possible to have >1 wall, combine first?
    """
    #cap_bounds = [None]*len(cap_id)
    wall = threshold_polydata(poly, 'ModelFaceID', (wall_id, wall_id))
    wall = subdivision(wall, num, mode)
    for i, c_id in enumerate(cap_id):
        cap = threshold_polydata(poly, 'ModelFaceID', (c_id, c_id))
        #cap_bounds[i] = find_boundary_edges(cap)
        cap = subdivision(cap, num, mode)
        wall = append_polydata(wall,cap)
        if clean:
            wall = clean_polydata(wall, 0.) 
    if clean:
        wall = clean_polydata(wall, 1e-5) 
    return wall

def subdivision(poly,num,option='linear'):

    if option == 'linear':
        divide = vtk.vtkLinearSubdivisionFilter()
    elif option == 'loop':
        divide = vtk.vtkLoopSubdivisionFilter()
    elif option == 'butterfly':
        divide = vtk.vtkButterflySubdivisionFilter()
    else:
        print("subdivision option: linear, loop or butterfly")
        raise
    divide.SetInputData(poly)
    divide.SetNumberOfSubdivisions(num)
    divide.Update()
    return divide.GetOutput()

def oriented_pointset_on_boundary(boundary):
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

def cap_polydata_openings(poly,  size):
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
    import os
    def _plot_points(points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        plt.show()
    def _add_nodes_to_cap(vtkPts, size):
        """
        Add uniformed points to cap
        """
        points = vtk_to_numpy(vtkPts.GetData())
        num = points.shape[0]
        #_plot_points(points)
        ctr = np.mean(points, axis=0)
        length = np.mean(np.linalg.norm(points-ctr, axis = 1))
        r = np.linspace(0.5*size/length, (length-size*0.8)/length,int(np.floor(length/size)))
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
        cleanedPts = clean_polydata(ptsPly, size*0.01)

        vtkPts.InsertPoints(vtkPts.GetNumberOfPoints()
                            ,cleanedPts.GetNumberOfPoints()
                            ,0
                            ,cleanedPts.GetPoints())
            
        #_plot_points(vtk_to_numpy(vtkPts.GetData()))
        return vtkPts

    def _delaunay_2d(vtkPts, boundary):
        """
        Delaunay 2D on input points
        """
        vtkPtsPly = vtk.vtkPolyData()
        vtkPtsPly.SetPoints(vtkPts)
        
        ids, pt_list = oriented_pointset_on_boundary(boundary)   

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
    poly = tag_polydata(poly, tag_id)

    edges = find_boundary_edges(poly)
    components = separate_disconnected_polydata(edges)
    id_lists = [None]*len(components)
    pt_lists = [None]*len(components)
    for i in range(len(id_lists)):
        id_lists[i] = find_point_correspondence(poly,components[i].GetPoints())
        pt_lists[i] = vtk.vtkPoints()
        pt_lists[i].DeepCopy(components[i].GetPoints())
        print('Found %d points for boundary %d\n' % (len(id_lists[i]),i))
   
    cap_pts_list = list()
    for boundary, ids, pts in zip(components, id_lists, pt_lists):
        cap_pts = _add_nodes_to_cap(pts, size)
        cap_pts_list.append(cap_pts)
        cap = _delaunay_2d(cap_pts, boundary)
        #cap = cutSurfaceWithPolygon(cap, boundary)
        #tag the caps
        tag_id +=1
        cap = tag_polydata(cap, tag_id)
        poly = append_polydata(poly, cap)
    
    #cap_pts_ids = list()
    #for cap_pts in cap_pts_list:
    #    cap_pts_ids.append(find_point_correspondence(poly,cap_pts))
    poly = fix_polydata_normals(poly)
    return poly

def get_polydata_volume(poly):
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

def threshold_polydata(poly, attr, threshold):
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
def convert_polydata_to_image_data(poly, ref_im, reverse=True):
    """
    Convert the vtk polydata to imagedata 

    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    # Have to copy to create a zeroed vtk image data
    ref_im_zeros = vtk.vtkImageData()
    ref_im_zeros.DeepCopy(ref_im)
    ref_im_zeros.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im_zeros.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im_zeros)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im_zeros)
    if reverse:
        stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    return output
