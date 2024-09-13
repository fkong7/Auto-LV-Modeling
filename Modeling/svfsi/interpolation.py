import os
import sys
sys.path.append(os.path.join(os.path.dirname(
__file__), "../src"))
import numpy as np
import matplotlib.pyplot as plt
import collections
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import io_utils
import utils
import re
import vtk
from scipy.interpolate import CubicSpline

"""
Functions to write interpolated surface meshes for perscribed wall motion

"""
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def cubic_spline_ipl(time, t_m, dt_m, boundary_queue):
    """
    Cubic Hermite spline interpolation for nodes movement
    see https://en.wikipedia.org/wiki/Cubic_Hermite_spline

    Args:
        time: time index in range(num_itpls)+1
        t_m: initial time point
        dt_m: number of iterpolations
        boundary_queue: list of VTK PolyData
    
    Returns:
        coords: coordinates of the interpolated mesh
    """

    boundary0 = vtk_to_numpy(boundary_queue[0].GetPoints().GetData())
    boundary1 = vtk_to_numpy(boundary_queue[1].GetPoints().GetData())
    boundary2 = vtk_to_numpy(boundary_queue[2].GetPoints().GetData())
    boundary3 = vtk_to_numpy(boundary_queue[3].GetPoints().GetData())

    dim = boundary0.shape[-1]

    t_a = (float(time) - t_m)/dt_m
    h_00 = 2*t_a*t_a*t_a - 3*t_a*t_a + 1
    h_10 = t_a*t_a*t_a - 2*t_a*t_a + t_a
    h_01 = - 2*t_a*t_a*t_a + 3*t_a*t_a
    h_11 = t_a*t_a*t_a - t_a*t_a

    v_m = (boundary2-boundary0)/dt_m/2
    v_m1 = (boundary3-boundary1)/dt_m/2
    coords = h_00*boundary1 + h_01*boundary2 + h_10*v_m*dt_m + h_11*v_m1*dt_m
    return coords

def find_index_in_array(x, y):
    """
    For x being a list containing y, find the index of each element of y in x
    """
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices

def move_mesh_cubic_hermite(meshes, start_point, intpl_num, num_cycle):
    total_num_phase = len(meshes)
    total_steps = total_num_phase * (intpl_num+1)*num_cycle
    initialized = False
    poly_template = meshes[start_point]
    ref_coords = vtk_to_numpy(poly_template.GetPoints().GetData())
    store = np.zeros((poly_template.GetNumberOfPoints(), 3, total_steps+1)) 
    count = 0
    # First cycle
    for msh_idx in list(range(start_point, total_num_phase))+ list(range(0, start_point)):
        if not initialized:
            boundary_queue = collections.deque(4*[None], 4)
            boundary_queue.append(meshes[(msh_idx+total_num_phase-1)%total_num_phase])
            boundary_queue.append(meshes[msh_idx])
            boundary_queue.append(meshes[(msh_idx+1)%total_num_phase])
            boundary_queue.append(meshes[(msh_idx+2)%total_num_phase])
            initialized = True
        else:
            boundary_queue.append(meshes[(msh_idx+2)%total_num_phase])

        for i_idx in range(intpl_num+1):
            new_coords = cubic_spline_ipl(i_idx, 0, intpl_num+1, boundary_queue)
            displacement = new_coords- ref_coords
            store[:, :, count] = displacement
            count+=1
    # The rest cycles are copies of first cycle
    for c in range(1,num_cycle):
        s = c*total_num_phase * (intpl_num+1)
        e = s + total_num_phase * (intpl_num+1)
        store[:,:,s:e] = store[:, :,0:count]
    
    return store

def move_mesh_cubic(meshes, start_point, intpl_num, num_cycle):
    total_num_phase = len(meshes)
    total_steps = total_num_phase * (intpl_num+1)*num_cycle
    poly_template = meshes[start_point]
    ref_coords = vtk_to_numpy(poly_template.GetPoints().GetData())
    store = np.zeros((poly_template.GetNumberOfPoints(), 3, total_steps+1)) 
    count = 0
    mesh_list = [meshes[idx] for idx in list(range(start_point, total_num_phase)) +\
            list(range(0, start_point))]
    mesh_list = [mesh_list[-3], mesh_list[-2], mesh_list[-1]] + mesh_list + \
            [mesh_list[0], mesh_list[1], mesh_list[2]]
    mesh_point_list = [vtk_to_numpy(m.GetPoints().GetData()) for m in mesh_list]
    mesh_points = np.array(mesh_point_list)
    time_list = [-3, -2, -1] + list(range(total_num_phase)) + \
            [total_num_phase, total_num_phase+1, total_num_phase+2]
    
    cs = CubicSpline(np.array(time_list), mesh_points)
    
    time_interp = np.linspace(0, total_num_phase, total_num_phase * (intpl_num+1))
    new_coords = cs(time_interp).transpose(1, 2, 0)
    displacements = new_coords - np.expand_dims(ref_coords, axis=-1)
    
    ## The rest cycles are copies of first cycle
    for c in range(num_cycle):
        s = c*total_num_phase * (intpl_num+1)
        e = s + total_num_phase * (intpl_num+1)
        store[:,:,s:e] = displacements
    return store

def write_motion(meshes,  start_point, intpl_num, output_dir, num_cycle, duration, debug=False, scale=1.):
    total_num_phase = len(meshes)
    total_steps = num_cycle* total_num_phase * (intpl_num+1)+1
    initialized = False
    time_pts = np.linspace(0,num_cycle*duration, total_steps)
    
    poly_template = meshes[start_point]
    
    displacements = move_mesh_cubic(meshes, start_point, intpl_num, num_cycle)
    #displacements = move_mesh_cubic_hermite(meshes, start_point, intpl_num, num_cycle)
    if debug:
        debug_dir = os.path.join(output_dir,"Debug")
        try:
            os.makedirs(debug_dir)
        except Exception as e: print(e)
        coords = vtk_to_numpy(poly_template.GetPoints().GetData())
        poly = vtk.vtkPolyData()
        poly.DeepCopy(poly_template)
        for ii in range(displacements.shape[-1]):
            poly.GetPoints().SetData(numpy_to_vtk(displacements[:,:,ii] * scale + coords * scale))
            fn_debug = os.path.join(debug_dir, "debug%05d.vtp" %ii)
            io_utils.write_vtk_polydata(poly, fn_debug)

    node_ids = vtk_to_numpy(poly_template.GetPointData().GetArray('GlobalNodeID'))
    face_ids = vtk_to_numpy(poly_template.GetCellData().GetArray('ModelFaceID'))
    #write time steps and node numbers
    for face in np.unique(face_ids):
        fn = os.path.join(output_dir, '%d_displacement.dat' % face)
        face_poly = utils.threshold_polydata(poly_template, 'ModelFaceID', (face,face))
        f = open(fn, 'w')
        f.write('{} {} {}\n'.format(3, total_steps,face_poly.GetNumberOfPoints()))
        for t in time_pts:
            f.write('{}\n'.format(t))
        #f.write('{}\n'.format(face_poly.GetNumberOfPoints()))
        face_ids = vtk_to_numpy(face_poly.GetPointData().GetArray('GlobalNodeID'))
        node_id_index = find_index_in_array(node_ids, face_ids)
        if debug:
            coords = vtk_to_numpy(face_poly.GetPoints().GetData())
            poly = vtk.vtkPolyData()
            poly.DeepCopy(face_poly)
            face_displacements = displacements[node_id_index, :, :] * scale
            for ii in range(face_displacements.shape[-1]):
                poly.GetPoints().SetData(numpy_to_vtk(face_displacements[:,:,ii]+coords*scale))
                fn_debug = os.path.join(debug_dir, "face{}debug{:05d}.vtp".format(face, ii))
                io_utils.write_vtk_polydata(poly, fn_debug)
        for i in node_id_index:
            disp = displacements[i, :, :] * scale
            f.write('{}\n'.format(node_ids[i]))
            for j in range(total_steps):
                f.write('{} {} {}\n'.format(disp[0,j], disp[1,j],disp[2,j]))
        f.close()




if __name__=='__main__':
    import time
    import argparse
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', help="Path to the surface meshes")
    parser.add_argument('--output_dir', help="Path to the volume meshes")
    parser.add_argument('--num_interpolation', type=int, help="Number of interpolations")
    parser.add_argument('--num_cycle', type=int, help="Number of cardiac cycles")
    parser.add_argument('--duration', type=float, help="Cycle duration in seconds")
    parser.add_argument('--scale', default=1., type=float, help="Scale displacements.")
    parser.add_argument('--mesh_complete_surface', default=None, help="Optional mesh complete exterior \
            file that can be used to make sure the correspondence between the simulation mesh input and \
            the motion meshes are correct")
    parser.add_argument('--phase', default=-1, type=int, help="Id of the phase to generate volume mesh")
    args = parser.parse_args()
    
    mesh_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, 'mesh-complete')

    try:
       os.makedirs(output_dir)
    except Exception as e: print(e)
    import glob
    fns = natural_sort(glob.glob(os.path.join(mesh_dir, "*.vtp")))
   
    meshes = [io_utils.read_vtk_mesh(f) for f in fns]

    if args.mesh_complete_surface is not None:
        print("!!!!!!!!", args.mesh_complete_surface)
        print("!!!!!!!!", fns[args.phase])
        surf_ori = io_utils.read_vtk_mesh(args.mesh_complete_surface)
        id_list = utils.find_point_correspondence(surf_ori, meshes[args.phase].GetPoints())
        # check if correspondence can be established
        points = vtk_to_numpy(meshes[args.phase].GetPoints().GetData())
        points = points[id_list, :]
        error = np.mean(np.linalg.norm(points - vtk_to_numpy(surf_ori.GetPoints().GetData()), axis=0))
        #surf_ori.GetPoints().SetData(numpy_to_vtk(points))
        if error > 1e-3:
            print("error: ", error)
            raise ValueError("There are uncorrected point mismatched between the motion files and mesh complete files")
        else:
            meshes[args.phase].GetPoints().SetData(surf_ori.GetPoints().GetData())
        for ind, m in enumerate(meshes):
            points = vtk_to_numpy(m.GetPoints().GetData())
            points_new = points[id_list, :]
            m = vtk.vtkPolyData()
            m.DeepCopy(surf_ori)
            m.GetPoints().SetData(numpy_to_vtk(points_new))
            meshes[ind] = m

    write_motion(meshes,  args.phase ,args.num_interpolation, output_dir, args.num_cycle, args.duration, debug=False, scale=args.scale)
    end = time.time()
    print("Time spent: ", end-start)
