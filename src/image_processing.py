import os
import numpy as np
import sys
import vtk
import utils
import label_io
import marching_cube as m_c


class Images(object):
    def __init__(self, fn):
        self.label = label_io.loadLabelMap(fn)
   
    def convert2binary(self):
        self.label = utils.convertVTK2binary(self.label)

    def resample(self, resolution, mode):
        self.label = utils.vtkImageResample(self.label, resolution, mode)

    def get_image(self):
        return self.label

    def generate_surface(self, region_id, smooth_iter):
        return m_c.vtk_marching_cube_multi(self.label, region_id, smooth_iter)

class lvImage(Images):
    
    def process(self, remove_list):
        self.label = utils.vtkImageResample(self.label, spacing=(0.5, 0.5, 0.5), opt='NN')
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        pylabel = vtk_to_numpy(self.label.GetPointData().GetScalars())
        #remove myocardium, RV, RA and PA
        for tissue in remove_list:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        self.label.GetPointData().SetScalars(numpy_to_vtk(pylabel))

    def buildCutter(self, region_id, adjacent_id, FACTOR, op='valve', smooth_iter=50):
        """
        Build cutter for aorta and la

        Args:
            label: original SimpleITK image
            op: 'valve' or 'tissue', option for normal direction
        """
        cut_Im = vtk.vtkImageData()
        cut_Im.DeepCopy(self.label)
        #locate centroid of mitral plane or aortic plane
        pts = utils.locateRegionBoundary(cut_Im, adjacent_id, region_id)
        ctr_valve = np.mean(pts, axis=0)
        #centroid of left atrium or aorta
        ctr = utils.getCentroid(cut_Im, region_id)
        #center and nrm of the cutting plane
        length = np.linalg.norm(ctr-ctr_valve)
        nrm_tissue = (ctr - ctr_valve)/length
        nrm_valve_plane = utils.fitPlaneNormal(pts)
        #check normal direction
        if op=='valve':
            nrm = nrm_valve_plane
        elif op=='tissue':
            nrm = nrm_tissue
        else:
            raise ValueError("Incorrect option")
        if np.dot(nrm_tissue, nrm_valve_plane)<0:
            nrm =  -1 *nrm
        ori = ctr_valve + length * FACTOR * nrm/np.linalg.norm(nrm)
    
        #dilate by a little bit
        cut_Im = utils.labelDilateErode(utils.recolorVTKPixelsByPlane(cut_Im, ori, -1.*nrm, 0), region_id, 0, 4)
        # marching cube
        cutter = m_c.vtk_marching_cube(cut_Im, region_id,smooth_iter)

        return cutter

