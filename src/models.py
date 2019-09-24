import os
import numpy as np
import sys
import vtk
import utils
import label_io
from sv import *

import meshing

from abc import ABCMeta, abstractmethod

class Geometry(object, metaclass=ABCMeta):
    pass

class leftVentricle(Geometry):
    
    def __init__(self, vtk_poly):
        self.poly = vtk_poly
    
    def processWall(self, la_cutter, aa_cutter):
        # cut with la and aorta cutter:
        self.poly = utils.cutPolyDataWithAnother(self.poly, la_cutter,False)
        self.poly = utils.cutPolyDataWithAnother(self.poly, aa_cutter,False)

        #improve valve opening geometry
        id_lists,boundaries = utils.getPointIdsOnBoundaries(self.poly)
        for idx, (ids, boundary) in enumerate(zip(id_lists, boundaries)):
            boundary = utils.smoothVTKPolyline(boundary, 2)
            self.poly = utils.projectOpeningToFitPlane(self.poly, ids, boundary.GetPoints(), 3)
            # Remove the free cells and update the point lists
            self.poly, id_lists[idx] = utils.removeFreeCells(self.poly, [idx for sub_l in id_lists for idx in sub_l])
        self.poly = utils.smoothVTKPolydata(utils.cleanPolyData(self.poly, 0.))
    
    def processCap(self, edge_size):
        self.poly, self.cap_pts_ids = utils.capPolyDataOpenings(self.poly, edge_size)

    def remesh(self, edge_size, fn, fns_out):

        # generate volumetric mesh:
        mesh_ops = {
                'SurfaceMeshFlag': True,
                'VolumeMeshFlag': True,
                'GlobalEdgeSize': edge_size, 
                'MeshWallFirst': True, 
                'NoMerge':True,
                'NoBisect': True,
                'Epsilon': 1e-8,
                'Optimization': 3,
                'QualityRatio': 1.4
        }
        poly_fn, ug_fn = meshing.meshPolyData(fn, fns_out, mesh_ops)
        self.poly = Repository.ExportToVtk(poly_fn)
        self.ug = Repository.ExportToVtk(ug_fn)
        return poly_fn, ug_fn 
   
    def update(self, new_model):

        # Project the cap points so that they are co-planar
        for pt_ids in self.cap_pts_ids:
            pts = utils.getPolyDataPointCoordinatesFromIDs(new_model, pt_ids)
            self.poly = utils.projectOpeningToFitPlane(new_model, pt_ids, pts, 3)

    def getVolume(self):
        return utils.getPolydataVolume(self.poly)

    def writeSurfaceMesh(self, fn):
        label_io.writeVTKPolyData(self.poly, fn)
    
    def writeVolumeMesh(self, fn):
        label_io.writeVTUFile(self.ug, fn)
