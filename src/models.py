import os
import numpy as np
import sys
import vtk
import utils
import label_io

class Geometry(object):
    def __init__(self, vtk_poly):
        self.poly = vtk_poly
    def getVolume(self):
        return utils.getPolydataVolume(self.poly)

    def writeSurfaceMesh(self, fn):
        label_io.writeVTKPolyData(self.poly, fn)
    
    def writeVolumeMesh(self, fn):
        label_io.writeVTUFile(self.ug, fn)
   
    def remesh(self, edge_size, fn, poly_fn=None, ug_fn=None):
        from sv import Repository
        import meshing
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
        if poly_fn is None:
            mesh_ops['SurfaceMeshFlag']=False
        if ug_fn is None:
            mesh_ops['VolumeMeshFlag']=False
        meshing.meshPolyData(fn, (poly_fn, ug_fn), mesh_ops)
        if poly_fn is not None:
            self.poly = Repository.ExportToVtk(poly_fn)
        if ug_fn is not None:
            self.ug = Repository.ExportToVtk(ug_fn)
        return poly_fn, ug_fn 

class leftVentricle(Geometry):
    
    def __init__(self, vtk_poly):
        super(leftVentricle, self).__init__(vtk_poly)
        self.wall_processed = False
        self.cap_processed = False
        self.cap_pts_ids = None

    def processWall(self, la_cutter, aa_cutter):
        if self.wall_processed:
            print("Left ventricle wall has been processed!")
            return
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

        self.wall_processed = True
        return

    def processCap(self, edge_size):
        if self.cap_processed:
            print("Caps have been processed!")
            return
        self.poly = utils.capPolyDataOpenings(self.poly, edge_size)
        self.cap_processed = True
        return

    def splitRegion(self, region_id):
        return utils.thresholdPolyData(self.poly, 'ModelFaceID', (region_id, region_id))

    def getCapIds(self):
        self.cap_pts_ids = list()
        # good to assume region id mitral=2, aortic=3
        for cap_id in (2,3):
            self.cap_pts_ids.append(utils.findPointCorrespondence(self.poly, self.splitRegion(cap_id).GetPoints()))
   
   
    def update(self, new_model):
        if self.cap_pts_ids is None:
            self.getCapIds()
        # Project the cap points so that they are co-planar
        for pt_ids in self.cap_pts_ids:
            pts = utils.getPolyDataPointCoordinatesFromIDs(new_model, pt_ids)
            new_model = utils.projectOpeningToFitPlane(new_model, pt_ids, pts, 3)
        return new_model

