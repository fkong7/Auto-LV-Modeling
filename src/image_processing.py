import os
import numpy as np
import sys
import vtk
import utils
import label_io
import marching_cube as m_c

#TO-DO improve compatibility with label ids, line 32
class Images(object):
    def __init__(self, fn):
        self.label = label_io.loadLabelMap(fn)
   
    def convert2binary(self):
        self.label = utils.convertVTK2binary(self.label)
        #self.label = utils.gaussianSmoothVTKImage(self.label, 0.01)

    def resample(self, resolution, mode):
        self.label = utils.vtkImageResample(self.label, resolution, mode)

    def get_image(self):
        return self.label
    
    def write_image(self,fn):
        label_io.writeVTKImage(self.label, fn)

    def generate_surface(self, region_id, smooth_iter, band):
        poly = m_c.vtk_marching_cube(self.label, region_id, smooth_iter, band)
        #return m_c.vtk_continuous_marching_cube(self.label, region_id, smooth_iter)
        return poly
class lvImage(Images):
    
    def process(self, remove_list):
        self.label = utils.vtkImageResample(self.label, spacing=(1.2, 1.2, 1.2), opt='NN')
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        pylabel = vtk_to_numpy(self.label.GetPointData().GetScalars())
        pylabel = utils.swapLabels(pylabel)
        #remove myocardium, RV, RA and PA
        for tissue in remove_list:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        self.label.GetPointData().SetScalars(numpy_to_vtk(pylabel))
        # remove connections between AA and LA
        self.label = utils.labelDilateErode(self.label, 6, 3, 8) #6 - AO id, 3 - LV id
        self.label = utils.labelOpenClose(self.label, 6, 0, size=5)
        self.label = utils.labelOpenClose(self.label, 0, 6, size=5)
        self.label = utils.labelDilateErode(self.label, 2, 3, 2) #6 - AO id, 3 - LV id
        self.label = utils.labelOpenClose(self.label, 2, 0, size=5)
        self.label = utils.labelOpenClose(self.label, 3, 0, size=5)
        self.label = utils.labelOpenClose(self.label, 0, 3, size=5)
        self.label = utils.labelOpenClose(self.label, 0, 2, size=5)
        ids = utils.locateRegionBoundaryIDs(self.label, 2, 6, size=3.)
        self.ids = np.vstack((ids, utils.locateRegionBoundaryIDs(self.label, 6, 2, size=6.)))
        #self.label = utils.labelOpenClose(self.label, 2, 0, size=5)
        self.label = utils.recolorVTKPixelsByIds(self.label, self.ids, 0)
    
    def buildCutter(self, region_id, avoid_id, adjacent_id, FACTOR, op='valve', smooth_iter=50):
        """
        Build cutter for aorta and la

        Args:
            label: original SimpleITK image
            region_id: id of aorta or LA to build cutter
            avoid_id: id of aorta or LA to avoid cutting into
            op: 'valve' or 'tissue', option for normal direction
        """
        cut_Im = vtk.vtkImageData()
        cut_Im.DeepCopy(self.label)
        #locate centroid of mitral plane or aortic plane
        pts = utils.locateRegionBoundary(cut_Im, adjacent_id, region_id, size=2.)
        ctr_valve = np.mean(pts, axis=0)
        
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        vtkpts = vtk.vtkPoints()
        vtkpts.SetData(numpy_to_vtk(pts))
        #centroid of left atrium or aorta
        ctr = utils.getCentroid(cut_Im, region_id)
        #center and nrm of the cutting plane
        length = np.linalg.norm(ctr-ctr_valve)
        nrm_tissue = (ctr - ctr_valve)/length
        nrm_valve_plane = utils.fitPlaneNormal(pts)
        print(nrm_valve_plane)
        #check normal direction
        if op=='valve':
            #nrm = nrm_valve_plane
            #if np.dot(nrm_tissue, nrm_valve_plane)<0:
            #    nrm =  -1 *nrm
            nrm = nrm_tissue
        elif op=='tissue':
            nrm = nrm_tissue
            #nrm = nrm_valve_plane
            #if np.dot(nrm_tissue, nrm_valve_plane)<0:
            #    nrm =  -1 *nrm
        else:
            raise ValueError("Incorrect option")
        ori = ctr_valve + FACTOR * nrm/np.linalg.norm(nrm)
        #dilate by a little bit
        cut_Im = utils.labelDilateErode(utils.recolorVTKPixelsByPlane(cut_Im, ori, -1.*nrm, 10, avoid_id), region_id, 0, 8.)
        cut_Im = utils.labelDilateErode(cut_Im, avoid_id, region_id, 2)
        
        # marching cube
        cutter = m_c.vtk_marching_cube(cut_Im, region_id)
        return cutter

