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

    def splitRegion(self, region_id, attr='ModelFaceID'):
        return utils.thresholdPolyData(self.poly, attr, (region_id, region_id))
    
    def remesh(self, edge_size, fn, poly_fn=None, ug_fn=None):
        from sv import Repository
        import meshing

        Repository.ImportVtkPd(self.poly, "mmg_poly")
        meshing.remeshPolyData("mmg_poly", "mmg_poly_remesh", 1.,1.5, fn)
        print("MMG GLOBAL REMESHING DONE")
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
    
    def writeMeshComplete(self, path):
        pass
    
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
        label_io.writeVTKPolyData(self.poly, '/Users/fanweikong/Downloads/test.vtp')
        self.poly = utils.cutPolyDataWithAnother(self.poly, la_cutter,False)
        self.poly = utils.cutPolyDataWithAnother(self.poly, aa_cutter,False)
        #fill small cutting artifacts:
        self.poly = utils.fillHole(self.poly, size=10)
        label_io.writeVTKPolyData(self.poly, '/Users/fanweikong/Downloads/test1.vtp')
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

    def writeMeshComplete(self, path):
        """
        Args: 
            path: path to the output folder
        """
        if (self.poly is None) or (self.ug is None):
            raise RuntimeError("No volume mesh has been generated.")
            return
        
        try:
            os.makedirs(os.path.join(path))
        except Exception as e: print(e)
        
        fn_poly = os.path.join(path, 'mesh-complete.exterior.vtp')
        fn_vol = os.path.join(path, 'mesh-complete.mesh.vtu')
        self.writeVolumeMesh(fn_vol)
        self.writeSurfaceMesh(fn_poly)

        fn_wall = os.path.join(path, 'walls_combined.vtp')
        label_io.writeVTKPolyData(self.splitRegion(1),fn_wall)
        try:
            os.makedirs(os.path.join(path, 'mesh-surfaces'))
        except Exception as e: print(e)

        for i in range(3):
            face = self.splitRegion(i+1)
            face_fn = os.path.join(path,'mesh-surfaces','noname_%d.vtp' % (i+1))
            label_io.writeVTKPolyData(face, face_fn)
        return

        


