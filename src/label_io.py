"""
IO functions for importing and exporting label maps and mesh surfaces

@author: Fanwei Kong

"""
import numpy as np
import os
import vtk

def loadLabelMap2Py(fn):
    """ 
    This function imports the label map as numpy arrays.

    Args: 
        fn: filename of the label map

    Return:
        label: numpy array of the label map
        spacing: spacing information of the label map
    """
    import SimpleITK as sitk
    mask = sitk.ReadImage(fn)
    label = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()

    return label, spacing

def loadLabelMap(fn):
    """ 
    This function imports the label map as sitk image.

    Args: 
        fn: filename of the label map

    Return:
        label: label map as a sitk image
    """
    import SimpleITK as sitk
    label = sitk.ReadImage(fn)

    return label

def exportPy2Sitk(npArr, sitkIm):
    """
    This function exports the numpy array to a sitk image
    
    Args:
        npArr: input numpy array
        sitkIm: reference sitk image
    Return:
        outSitkIm: output sitk image

    """
    import SimpleITK as sitk
    outSitkIm = sitk.GetImageFromArray(npArr)
    outSitkIm.SetSpacing(sitkIm.GetSpacing())
    outSitkIm.SetOrigin(sitkIm.GetOrigin())
    outSitkIm.SetDirection(sitkIm.GetDirection())
    return outSitkIm

def writeSitkIm(sitkIm, fn):
    """
    This function writes a sitk image to disk
    Args:
        sitkIm: the sitk image to write
        fn: file name
    Return:
        None

    """
    import SimpleITK as sitk
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fn)
    writer.Execute(sitkIm)

    return

def isoSurf2VTK(verts, faces):
    """
    This function creates a polydata from numpy arrays

    Args:
        verts: vertice coordinates of the isosurface
        faces: connectivity
    Returns:
        poly: VTK polydata

    """
    vtkPts = vtk.vtkPoints()
    vtkPts.SetNumberOfPoints(verts.shape[0])
    for i in range(verts.shape[0]):
        vtkPts.SetPoint(i, verts[i,:])

    vtkCells = vtk.vtkCellArray()
    #vtkCells.SetNumberOfCells(faces.shape[0])
    for i in range(faces.shape[0]):
        if faces.shape[-1]==3:
            cell = vtk.vtkTriangle()
        else:
            raise ValueError('Inconsistent number of face id')
        for j in range(faces.shape[-1]):
            cell.GetPointIds().SetId(j, faces[i,j])

        vtkCells.InsertNextCell(cell)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtkPts)
    poly.SetPolys(vtkCells)
    def _fixNormals(poly):
        normAdj = vtk.vtkPolyDataNormals()
        normAdj.SetInputData(poly)
        normAdj.SplittingOff()
        normAdj.ConsistencyOn()
        normAdj.FlipNormalsOn()
        normAdj.Update()
        poly = normAdj.GetOutput()
        return poly

    poly = _fixNormals(poly)
    return poly

def writeVTKPolyData(poly, fn):
    """
    This function writes a vtk polydata to disk
    Args:
        poly: vtk polydata
        fn: file name
    Returns:
        None
    """
    
    print('Writing vtp with name:', fn)
    if (fn == ''):
        return 0

    _ , extension = os.path.splitext(fn)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError("Incorrect extension"+extension)
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def writeVTKImage(vtkIm, fn):
    """
    This function writes a vtk image to disk
    Args:
        vtkIm: the vtk image to write
        fn: file name
    Returns:
        None
    """
    print("Writing vti with name: ", fn)

    _, extension = os.path.splitext(fn)
    if extension == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif extension == '.mhd':
        writer = vtk.vtkMetaImageWriter()
    else:
        raise ValueError("Incorrect extension " + extension)
    writer.SetInputData(vtkIm)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array

    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
    
    #vtkArray = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    vtkArray = numpy_to_vtk(img.transpose(0,1,2).flatten())
    return vtkArray


def exportSitk2VTK(sitkIm):
    """
    This function creates a vtk image from a simple itk image

    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    import SimpleITK as sitk
    img = sitk.GetArrayFromImage(sitkIm)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin(sitkIm.GetOrigin())
    imageData.SetSpacing(sitkIm.GetSpacing())
    #imageData.SetDirectionMatrix(sitkIm.GetDirection()) 
    
    return imageData

def loadVTKMesh(fileName):
    """
    Loads surface/volume mesh to VTK
    """
    if (fileName == ''):
        return 0
    fn_dir, fn_ext = os.path.splitext(fileName)
    if (fn_ext == '.vtk'):
        print('Reading vtk with name: ', fileName)
        reader = vtk.vtkPolyDataReader()
    elif (fn_ext == '.vtp'):
        print('Reading vtp with name: ', fileName)
        reader = vtk.vtkXMLPolyDataReader()
    elif (fn_ext == '.stl'):
        print('Reading stl with name: ', fileName)
        reader = vtk.vtkSTLReader()
    elif (fn_ext == '.vtu'):
        print('Reading vtu with name: ', fileName)
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif (fn_ext == '.pvtu'):
        print('Reading pvtu with name: ', fileName)
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        raise ValueError('File extension not supported')

    reader.SetFileName(fileName)
    reader.Update()
    return reader.GetOutput()

def writeVTUFile(ug, fn):
    print('Writing vts with name:', fn)
    if (fn == ''):
        raise ValueError('File name is empty')
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(ug)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()

def writePointCloud(pts,fn):
    """
    Write VTK points to Elastix point format
    """
    with open(fn,'w') as f:
        f.write('point\n')
        f.write('%d\n' % pts.GetNumberOfPoints())
        for i in range(pts.GetNumberOfPoints()):
            pt = pts.GetPoint(i)
            f.write('%f %f %f\n' % (pt[0], pt[1], pt[2]))

    return

    
    

def writeVTKPolyDataVerts(poly, fn):
    """
    Writes the vertices of the VTK PolyData
    """
    print('Writing pts with name: ', fn)
    pts = poly.GetPoints()
    writePointCloud(pts, fn)
    return


def readElastixPointOuptut(fn):
    """
    Read the point coordinates after registration generated by Elastix

    Args: 
        fn: file name
    Returns:
        pts: vtk 
    """
    import re
    pts = vtk.vtkPoints()
    with open(fn, 'r') as f:
        for line in f:
            s = re.findall(r'[+-]?\d+(?:\.\d+)?', line)
            if len(s)>0:
                s = s[10:13]
                pts.InsertNextPoint([float(i) for i in s])

    print('Reading %d points from file %s' % (pts.GetNumberOfPoints(), fn))
    return pts

