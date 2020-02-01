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

    def resample(self, resolution, mode):
        self.label = utils.vtkImageResample(self.label, resolution, mode)

    def get_image(self):
        return self.label
    
    def write_image(self,fn):
        label_io.writeVTKImage(self.label, fn)

    def generate_surface(self, region_id, smooth_iter):
        return m_c.vtk_marching_cube_multi(self.label, region_id, smooth_iter)

class lvImage(Images):
    
    def process(self, remove_list):
        #self.write_image('/Users/fanweikong/Downloads/test0.vti') 
        self.label = utils.vtkImageResample(self.label, spacing=(1.2, 1.2, 1.2), opt='NN')
        self.write_image('/Users/fanweikong/Downloads/test1.vti') 
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        pylabel = vtk_to_numpy(self.label.GetPointData().GetScalars())
        pylabel = utils.swapLabels(pylabel)
        #remove myocardium, RV, RA and PA
        for tissue in remove_list:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        self.label.GetPointData().SetScalars(numpy_to_vtk(pylabel))
        self.write_image('/Users/fanweikong/Downloads/test2.vti') 
        # remove connections between AA and LA
        ids = utils.locateRegionBoundaryIDs(self.label, 2, 6, size=3.)
        ids = np.vstack((ids, utils.locateRegionBoundaryIDs(self.label, 6, 2, size=6.)))
        self.label = utils.recolorVTKPixelsByIds(self.label, ids, 0)
        self.write_image('/Users/fanweikong/Downloads/test3.vti') 
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
        pts = utils.locateRegionBoundary(cut_Im, adjacent_id, region_id, size=4.)
        ctr_valve = np.mean(pts, axis=0)
        
        print("-----------------------")
        print("CTR_VALVE: ", ctr_valve)
        
        #centroid of left atrium or aorta
        ctr = utils.getCentroid(cut_Im, region_id)
        print("CTR: ", ctr)
        #center and nrm of the cutting plane
        length = np.linalg.norm(ctr-ctr_valve)
        nrm_tissue = (ctr - ctr_valve)/length
        nrm_valve_plane = utils.fitPlaneNormal(pts)
        #check normal direction
        if op=='valve':
            nrm = nrm_valve_plane
            if np.dot(nrm_tissue, nrm_valve_plane)<0:
                nrm =  -1 *nrm
        elif op=='tissue':
            nrm = nrm_tissue
        else:
            raise ValueError("Incorrect option")
        ori = ctr_valve + length * FACTOR * nrm/np.linalg.norm(nrm)
        print("ORI: ", ori)
        print("NRM: ", nrm)
        print("----------------------")
        #dilate by a little bit
        cut_Im = utils.labelDilateErode(utils.recolorVTKPixelsByPlane(cut_Im, ori, -1.*nrm, 10, avoid_id), region_id, 0, 8.)
        cut_Im = utils.labelDilateErode(cut_Im, avoid_id, region_id, 2)
        debug_fn = '/Users/fanweikong/Downloads/cut_'+str(region_id) + '.vti'
        label_io.writeVTKImage(cut_Im, debug_fn)
        
        # marching cube
        cutter = m_c.vtk_marching_cube(cut_Im, region_id,smooth_iter)
        return cutter

