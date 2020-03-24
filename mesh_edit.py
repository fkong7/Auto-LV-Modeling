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
    poly_fn = '/Users/fanweikong/Documents/Modeling/pycpd/data/meshes/multi_dataset/ct_train_1010_label.nii.gz.vtk'
    poly_tmplt_fn = '/Users/fanweikong/Documents/Modeling/pycpd/data/meshes/ct_train_10_lvcfd_tmplt.vtp'
    poly = label_io.loadVTKMesh(poly_fn)
    poly_tmplt =label_io.loadVTKMesh(poly_tmplt_fn)
    label_io.writeVTKPolyData(poly_tmplt, poly_tmplt_fn)

    print(poly_tmplt.GetNumberOfPoints())
    poly_tmplt = insertPtByNodeId(poly, poly_tmplt, 1606)
    print("ok")
    print(poly_tmplt.GetNumberOfPoints())
    poly_tmplt = insertTriByNodeIds(poly_tmplt, poly_tmplt.GetNumberOfPoints()-1, 36, 2)
    
    label_io.writeVTKPolyData(poly_tmplt, os.path.join(os.path.dirname(poly_tmplt_fn), 'tmplt.vtp'))
    label_io.writeVTKPoints(poly_tmplt.GetPoints(), os.path.join(os.path.dirname(poly_tmplt_fn), 'tmplt_pts.vtp'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="", help="Input folder name")
    parser.add_argument('--output',default="", help="Output folder name")
    parser.add_argument('--rate', type=float, default=0.8, help="Target reduction rate")
    args = parser.parse_args()
    #decimate_mesh_in_folder(args.input, args.output, args.rate)
    build_template()
