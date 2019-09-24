import os
import numpy as np
import sys
import vtk
import utils
from sv import *

from abc import ABCMeta, abstractmethod

class Images(object, metaclass=ABCMeta):
    pass

class lvImage(Images):

    def __init__(self, fn):
        self.label = label_io.loadLabelMap(fn)

    def process(self):
        label = utils.resample(label)
        pylabel = sitk.GetArrayFromImage(label)
        #remove myocardium, RV, RA and PA
        for tissue in [1, 4, 5,7]:
            pylabel = utils.removeClass(pylabel, tissue, 0)
        vtkIm = label_io.exportSitk2VTK(label_io.exportPy2Sitk(pylabel, label))
        
        def _buildCutter(label,region_id, adjacent_id, FACTOR, op='valve'):
            """
            Build cutter for aorta and la

            Args:
                label: original SimpleITK image
                op: 'valve' or 'tissue', option for normal direction
            """
            cut_Im = label_io.exportSitk2VTK(label)
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
            cutter = m_c.vtk_marching_cube(cut_Im, region_id,50)

            return cutter
