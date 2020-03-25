import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import glob
import numpy as np
import vtk
import label_io
import utils
import argparse

def decimate_mesh_in_folder(dir_n, out_n, rate):
    """
    Decimate the meshes in a folder with the same target reduction rate
    Args:
        dir_n: folder name
        rate: tagret reduction rate
        out_n: output folder name
    """
    fns = glob.glob(os.path.join(dir_n, '*.vtp'))+glob.glob(os.path.join(dir_n, '*.vtk'))
    try:
        os.makedirs(out_n)
    except Exception as e: print(e)
    
    for fn in fns:
        poly = label_io.loadVTKMesh(fn)
        poly = utils.decimation(poly, rate)
        poly_n = os.path.basename(fn)
        label_io.writeVTKPolyData(poly, os.path.join(out_n, poly_n))

def insertTriByNodeIds(poly,id1,id2,id3):
    tri = vtk.vtkTriangle()
    tri.GetPointIds().SetId(0,id1)
    tri.GetPointIds().SetId(1,id2)
    tri.GetPointIds().SetId(2,id3)
    cells = poly.GetPolys()
    cells.InsertNextCell(tri)
    poly.SetPolys(cells)
    print("ok1") 
    #make sure normals are correct
    #return utils.fixPolydataNormals(poly)
    return poly

def insertPtByNodeId(source, target, pt_id):
    """
    source: the mesh to retrieve point coordinates
    target: the mesh to insert new point
    pt_id: node id on source mesh
    """
    pt = source.GetPoints().GetPoint(pt_id)
    target.GetPoints().InsertNextPoint(pt)
    return target

def build_template():
    poly_fn = '/Users/fanweikong/SurfaceModeling/template/ct_train_1010_label.nii.gz.vtk'
    poly_tmplt_fn = '/Users/fanweikong/SurfaceModeling/template/ct_train_10_lvcfd_tmplt.vtp'
    poly = label_io.loadVTKMesh(poly_fn)
    poly_tmplt =label_io.loadVTKMesh(poly_tmplt_fn)
    label_io.writeVTKPolyData(poly_tmplt, poly_tmplt_fn)

    #NODE Betwen AV and MV
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3324)
    new_id = poly_tmplt.GetNumberOfPoints() - 1

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 317)
    first_id = poly_tmplt.GetNumberOfPoints() - 1
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 30, 36)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 314)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 36)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 36, 2)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3708)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 2)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 2, 43)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4510)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 43)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 43, 39)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4190)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 39)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 39, 6)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4343)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 6)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 6, 8)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4222)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 8)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 8, new_id) 
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3046)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, new_id)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, new_id, 30)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, first_id, poly_tmplt.GetNumberOfPoints()-1, 30)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1008)
    first_id = poly_tmplt.GetNumberOfPoints()-1
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 48, 49)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4234)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 49)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 49, 50)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3475)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 50)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 50, 51)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2329)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 51)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 51, 52)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4540)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 52)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 52, 53)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3628)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 53)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 53, 54)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2967)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 54)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 54, 47)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1378)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 47)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 47, 48)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, first_id, poly_tmplt.GetNumberOfPoints()-1, 48)
    
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 279)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 42, 27)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 27, 30)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2688)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 30)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 30, new_id)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3720)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 30, new_id)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, new_id)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, new_id, 8)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3300)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 8)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 8, 20)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2923)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 20, 42)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 20)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3767)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 67, 42)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2754)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 42)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 591)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 42)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 42, 63)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 615)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 63)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 63, 64)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 468)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 64)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 64, 65)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3488)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 65)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 65, 66)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 66, 67)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 67, 68)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3467)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 68, 69)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1876)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 69)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 69, 70)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 883)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 70)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 70, 71)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 166)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 71)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 71, 72)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1878)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 72)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 72, 73)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3583)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 73)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 73, 68)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 68, 74)
    
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 2735)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 74, 75)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1342)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 75, 76)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 75)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 740)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 76, 77)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 76)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 881)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 77, 78)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 77)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1942)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 78, 79)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 78)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 3275)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 79, 74)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 79)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 74, 80)

    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4399)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 85, 80)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1669)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 80, 81)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 80)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 780)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 81, 82)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 81)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 655)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 82, 83)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 82)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 704)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 83, 84)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 83)
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 4593)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 84, 85)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, poly_tmplt.GetNumberOfPoints()-2, 84)
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 85, 86)

    
    label_io.writeVTKPolyData(poly_tmplt, os.path.join(os.path.dirname(poly_tmplt_fn), 'tmplt.vtp'))
    label_io.writeVTKPoints(poly_tmplt.GetPoints(), os.path.join(os.path.dirname(poly_tmplt_fn), 'tmplt_pts.vtp'))

def mean_template(fn_d):
    fns = glob.glob(os.path.join(fn_d, '*.vtp'))+glob.glob(os.path.join(fn_d, '*.vtk'))
    poly = label_io.loadVTKMesh(fns[0])
    coords = np.zeros((poly.GetNumberOfPoints(),3))
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    for fn in fns:
        ply = label_io.loadVTKMesh(fn)
        coords_ply = vtk_to_numpy(ply.GetPoints().GetData())
        coords_ply -= np.mean(coords_ply, axis=0)
        coords += coords_ply

    coords /= len(fns)
    poly.GetPoints().SetData(numpy_to_vtk(coords))
    label_io.writeVTKPolyData(poly, os.path.join(fn_d, 'mean.vtp'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="", help="Input folder name")
    parser.add_argument('--output',default="", help="Output folder name")
    parser.add_argument('--rate', type=float, default=0.8, help="Target reduction rate")
    args = parser.parse_args()
    #decimate_mesh_in_folder(args.input, args.output, args.rate)
    #build_template()
    mean_template('/Users/fanweikong/Documents/Modeling/pycpd/data/registered/multidataset_1_1000')
