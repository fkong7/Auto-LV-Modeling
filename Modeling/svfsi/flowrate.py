import vtk
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import label_io
import utils
def convertPointDataToCellData(mesh):
    p2c = vtk.vtkPointDataToCellData()
    p2c.SetInputData(mesh)
    p2c.Update()
    return p2c.GetOutput()

def extracSurface(volume):
    extractor = vtk.vtkDataSetSurfaceFilter()
    extractor.SetInputData(volume)
    extractor.SetPieceInvariant(True)
    extractor.Update()
    return extractor.GetOutput()

def computeNormals(poly,angle):
    normalGen = vtk.vtkPolyDataNormals()
    normalGen.SetInputData(poly)
    normalGen.ComputeCellNormalsOn()
    normalGen.SetFeatureAngle(angle)
    #normalGen.SplittingOff()
    normalGen.Update()
    poly = normalGen.GetOutput()
    return poly

def cellArea(cell):
    pt1 = [0,0,0]
    pt2 = [0,0,0]
    pt3 = [0,0,0]
    cell.GetPoints().GetPoint(0,pt1)
    cell.GetPoints().GetPoint(1,pt2)
    cell.GetPoints().GetPoint(2,pt3)
    area = vtk.vtkTriangle.TriangleArea(pt1,pt2,pt3)
    return area

def extractRegions(poly):
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.ColorRegionsOn()
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    num_surf = connectivity.GetNumberOfExtractedRegions()
    print('Number of extracted surfaces: %d'%num_surf)
    extracted_regions = connectivity.GetOutput()
    return extracted_regions

def findCellsByRegion(poly,rg):
    List = []
    regionIds = poly.GetCellData().GetAbstractArray('RegionId')
    if regionIds == None:
        raise ValueError('Regions have not been extracted\n')
    for i in range(poly.GetNumberOfCells()):
        ID = regionIds.GetTuple(i)
        if int(ID[0]) == rg:
            List.append(i)

    return List

def findCellsByPoints(poly, ptIds):
    """ 
    Based on id number of points on a face, find the id of the cells on that face
    """
    List = []
    for p_id in ptIds:
        cell_list = vtk.vtkIdList()
        poly.GetPointCells(p_id, cell_list)
        for c_id in range(cell_list.GetNumberOfIds()):
            pt_list = vtk.vtkIdList()
            poly.GetCellPoints(cell_list.GetId(c_id), pt_list)
            on_face = True
            for pp_id in range(pt_list.GetNumberOfIds()):
                if pt_list.GetId(pp_id) not in ptIds:
                    on_face = False
            if on_face:
                if cell_list.GetId(c_id) not in List:
                    List.append(cell_list.GetId(c_id))
    return List

def setupSurfaceMesh(fileName):
    volMesh = label_io.loadVTKMesh(fileName)
    poly = extracSurface(volMesh)
    #polyCD = computeNormals(polyCD,40)

    return poly

def flowRate(polyCD,IdList):
    #get velocity data
    polyCD = computeNormals(polyCD, 20)
    polyCD = convertPointDataToCellData(polyCD)
    velCells = polyCD.GetCellData()
    norms = velCells.GetNormals()
    velArray = velCells.GetAbstractArray('Velocity')
    Q = 0.
    A = 0.
    count = 0
    for i in IdList:
        cell = polyCD.GetCell(i)
        area = cellArea(cell)
        vel = velArray.GetTuple(i)
        norm = norms.GetTuple(i)
        q = np.dot(np.array(vel),np.array(norm))*area*1e4
        Q = Q+q
        A = A + area
        count = count+1
        
    #print('Flow rate is %f mL/s' % Q)
    #print('Total number of cells in region is %d' % count)
    return Q

#def getMaxVelocity(mesh, IdList):
#    velPts = mesh.GetPointData().GetAbstractArray('Velocity')
#    from vtk.util.numpy_support import vtk_to_numpy
#    py_array = vtk_to_numpy(velPts)
#    #return np.percentile(py_array, 99)
#    return np.max(py_array)
def getMaxVelocity(mesh, IdList):
    velPts = mesh.GetPointData().GetAbstractArray('Velocity')
    maxVel = 0.
    for i in IdList:
        vel = np.linalg.norm(velPts.GetTuple(i))
        if abs(vel)>maxVel:
            maxVel = vel

    return maxVel

def getAllMaxVelocity(fns, face_poly_fn):
    polyCD = setupSurfaceMesh(fns[0])

    face_poly = label_io.loadVTKMesh(face_poly_fn)
    face_pts = face_poly.GetPoints()
    
    pt_ids = utils.findPointCorrespondence(polyCD, face_pts)
    
    Vlist = []
    for fn in fns:
        #if mode=='mv':
        #    IdList = mvList
        #elif mode=='av':
        #    IdList = avList
        poly = setupSurfaceMesh(fn)
        V = getMaxVelocity(poly, pt_ids)/1e3
        if not np.isnan(V):
            Vlist.append(V)
    return Vlist

def getAllVolume(fns):
    
    Vlist = []
    for fn in fns:
        poly = setupSurfaceMesh(fn)
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        Vlist.append(mass.GetVolume()*1.e-3)
    return Vlist

def getAllFlowRate(fns, face_poly_fn):
    polyCD = setupSurfaceMesh(fns[0])

    face_poly = label_io.loadVTKMesh(face_poly_fn)
    face_pts = face_poly.GetPoints()
    
    pt_ids = utils.findPointCorrespondence(polyCD, face_pts)
    IdList = findCellsByPoints(polyCD, pt_ids)
    tags = vtk.vtkIntArray()
    tags.SetNumberOfComponents(1)
    tags.SetName('Region Ids')
    tags.SetNumberOfValues(polyCD.GetNumberOfCells())
    for i in range(polyCD.GetNumberOfCells()):
        if i in IdList:
            tags.InsertValue(i, 2)
        else:
            tags.InsertValue(i, 1)
    polyCD.GetCellData().SetScalars(tags)
    label_io.writeVTKPolyData(polyCD, '/Users/fanweikong/Downloads/'+str(np.random.randint(100))+'.vtp')
    #polyCD = extractRegions(polyCD)
    #avList = findCellsByRegion(polyCD,2)
    #mvList = findCellsByRegion(polyCD,1)
    Qlist = []
    for fn in fns:
        #if mode=='mv':
        #    IdList = mvList
        #elif mode=='av':
        #    IdList = avList
        poly = setupSurfaceMesh(fn)
        Q = -1. * flowRate(poly, IdList)/1e7
        if not np.isnan(Q):
            Qlist.append(Q)
    return Qlist

import re
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def main():
    #dir_name = '/Volumes/Untitled/LVFSI_data/MACS40282_20150504_results_coarse'
    #dir_name_gt = '/Volumes/Untitled/LVFSI_data/MACS40282_20150504_gt_results_coarse2'
    dir_name = '/Volumes/Untitled/LVFSI_data/MACS40244_20150309_results_coarse2'
    dir_name_gt = '/Volumes/Untitled/LVFSI_data/MACS40244_20150309_gt_results_coarse2'
    face = {}
    face['av'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results/MACS40244_20150309/MACS40244_20150309-mesh-complete-coarse/mesh-surfaces/noname_3.vtp'
    face['mv'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results/MACS40244_20150309/MACS40244_20150309-mesh-complete-coarse/mesh-surfaces/noname_2.vtp'
    face_gt = {}
    face_gt['av'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results_gt/MACS40244_20150309_gt/MACS40244_20150309_gt-mesh-complete-coarse/mesh-surfaces/noname_3.vtp'
    face_gt['mv'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results_gt/MACS40244_20150309_gt/MACS40244_20150309_gt-mesh-complete-coarse/mesh-surfaces/noname_2.vtp'
    #face['av'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results/MACS40282_20150504/MACS40282_20150504-mesh-complete-coarse/mesh-surfaces/noname_3.vtp'
    #face['mv'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results/MACS40282_20150504/MACS40282_20150504-mesh-complete-coarse/mesh-surfaces/noname_2.vtp'
    #face_gt['av'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results_gt2/MACS40282_20150504/MACS40282_20150504_gt-mesh-complete-coarse/mesh-surfaces/noname_3.vtp'
    #face_gt['mv'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based_results_gt2/MACS40282_20150504/MACS40282_20150504_gt-mesh-complete-coarse/mesh-surfaces/noname_2.vtp'
    prefix = ['result_1_', 'result_2_', 'result_3_']
    mode = ['mv', 'av', 'mv']
    #prefix = ['result_2_']
    #mode = ['av']
    Qlist = []
    Qlist_gt = []
    for pre, m in zip(prefix, mode):
        fns = natural_sort(glob.glob(os.path.join(dir_name, pre+'*.vtu')))
        fns = [fns[i] for i in range(0, len(fns), 10)]
        fns_gt = natural_sort(glob.glob(os.path.join(dir_name_gt, pre+'*.vtu')))
        fns_gt = [fns_gt[i] for i in range(0, len(fns_gt), 10)]
        
        num_file = len(fns) if len(fns) < len(fns_gt) else len(fns_gt)
        fns = fns[:num_file]
        fns_gt = fns_gt[:num_file]
        
        res = getAllFlowRate(fns, face[m])
        Qlist += res
        #res_gt = getAllFlowRate(fns_gt, face_gt[m])
        #Qlist_gt += res_gt
        #res = getAllMaxVelocity(fns, face[m])
        #res_gt = getAllMaxVelocity(fns_gt, face_gt[m])
        #if m == 'av':
        #    Qlist += res
        #    Qlist_gt += res_gt
        #elif m=='mv':
        #    Qlist += [-1.*r for r in res] 
        #    Qlist_gt += [-1.*r for r in res_gt]
        #Qlist += getAllVolume(fns)
        #Qlist_gt += getAllVolume(fns_gt)
        #res = np.array(res)
        #res_gt = np.array(res_gt)

        #diff = np.abs(res_gt-res)
        #max_id = np.argmax(diff)
        #print(pre, m, np.max(np.abs(res)), np.min(np.abs(res)), diff[max_id]/np.abs(res_gt[max_id]))
    
    
    time = np.linspace(0, 1, len(Qlist))

    plt.rcParams.update({'font.size': 20})
    plt.plot(time, Qlist, '-', linewidth=3)
    #plt.plot(time, Qlist_gt, '-', linewidth=3)
    plt.xlabel('Time')
    plt.ylabel('Flow rate (ml/s)')

    #plt.legend(('Automated','Ground Truth'),loc='upper right')
    #plt.title('flow rate (ml/s)')
    plt.show()
    #plt.savefig(prefix+'flowRate.png')



if __name__ == '__main__':
    main()

