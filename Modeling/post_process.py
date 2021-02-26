import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import utils
from label_io import loadVTKMesh, writeVTKPolyData
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import argparse 


def create_coplanar_openings(model, cap_id_list, edge_size=1.):
    new_model = vtk.vtkPolyData()
    new_model.DeepCopy(model)
    p_ids_list = []
    for c_id in cap_id_list:
        print(c_id)
        cap = utils.thresholdPolyData(new_model, 'ModelFaceID', (c_id, c_id))
        p_ids = utils.findPointCorrespondence(new_model, cap.GetPoints())
        p_ids_list += p_ids
        new_model = utils.projectOpeningToFitPlane(new_model, p_ids, cap.GetPoints(), edge_size)
    
    for index in range(new_model.GetNumberOfPoints()):
        if index in p_ids_list:
            continue
        pt = np.array(new_model.GetPoints().GetPoint(index))
        ptList=vtk.vtkIdList()
        tempList=vtk.vtkIdList()
        tempPtList=vtk.vtkIdList()
        tempCoord=np.zeros((0,3))
        new_model.GetPointCells(index,tempList)
        for i in range(tempList.GetNumberOfIds()):
            new_model.GetCellPoints(tempList.GetId(i),tempPtList)
            for j in range(3):
                if tempPtList.GetId(j)!=index:
                    ptList.InsertUniqueId(tempPtList.GetId(j))
        for i in range(ptList.GetNumberOfIds()):
            tempCoord = np.vstack((tempCoord, np.array(new_model.GetPoints().GetPoint(ptList.GetId(i)))))
        mean_pt = np.sum(tempCoord, axis=0)/tempCoord.shape[0]
        new_model.GetPoints().SetPoint(index, mean_pt)
    return new_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Path to the model directory')
    parser.add_argument('--output_dir', help='Path to the output directory')
    parser.add_argument('--cap_ids', nargs='+', type=int, help='ModelFaceID of the caps, generated from SV')
    parser.add_argument('--edge_size', type=float, help='Maximum edge size of the surface mesh')
    args = parser.parse_args()
    try:
        os.makedirs(args.output_dir)
    except Exception as e:
        print(e)
    fns = glob.glob(os.path.join(args.input_dir, '*.vtp'))
    for fn in fns:
        name = os.path.basename(fn)
        out_fn = os.path.join(args.output_dir, name)
        model = loadVTKMesh(fn)
        processed = create_coplanar_openings(model, args.cap_ids, args.edge_size)
        writeVTKPolyData(processed, out_fn)
    
