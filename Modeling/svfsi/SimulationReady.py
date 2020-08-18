import numpy as np 
from vtk.util import numpy_support as vtknp

def read_vtp(input_surf):
    '''Read vtk and vtp data'''
    import numpy as np
    import vtk
    from vtk.util import numpy_support as vtknp

    if input_surf.endswith('vtk'):
        reader = vtk.vtkPolyDataReader()
    elif input_surf.endswith('vtp'):
        reader = vtk.vtkXMLPolyDataReader()

    reader.SetFileName(input_surf)
    reader.Update()
    polydata = reader.GetOutput()
    Npts     = reader.GetNumberOfPoints()

    # coordinates of points
    temp_vtk_array = polydata.GetPointData().GetArray("GlobalNodeID")
    GNID = vtknp.vtk_to_numpy(temp_vtk_array)
    temp_vtk_array = polydata.GetPoints().GetData()
    coord = vtknp.vtk_to_numpy(temp_vtk_array)

    return coord, GNID, Npts

def write_motion(Path,partname):
    fname = Path + partname + '.vtp'
    print('Reading '+'fname')
    coord, GlobalNodeId, NPoint = read_vtp(fname)
    print(NPoint)
    fout = partname + '_Motion.dat'
    fobj=open(fout,'w')
    fobj.write('{} {} {}\n'.format(3,NumT,NPoint))
    for i in range(NumT):
        fobj.write('{}\n'.format(Time[i]))
    
    for i in range(NPoint):
        fobj.write('{}\n'.format(GlobalNodeId[i]))
        k = G2L[GlobalNodeId[i]]
        for j in range(NumT):
            fobj.write('{} {} {}\n'.format(dispx[k,j],dispy[k,j],dispz[k,j]))

    fobj.close()
    return 


NPoint = 4754
NumT = 3001

dispx = np.zeros((NPoint,NumT),dtype=np.float)
dispy = np.zeros((NPoint,NumT),dtype=np.float)
dispz = np.zeros((NPoint,NumT),dtype=np.float)
Count = 0
for NN in range(NumT-1):
    i = NN + 1
    fname = './Interpolated/'+"mesh-complete.exterior_"+format(i, '05d')+'.txt'
    print(fname)
    results = np.loadtxt(fname)
    x = results[:,0]#[:NPoint]
    y = results[:,1]#[:NPoint]
    z = results[:,2]#[:NPoint]

    if NN == 0:
        xref = np.copy(x) 
        yref = np.copy(y)
        zref = np.copy(z)

    dispx[:,Count] = x - xref
    dispy[:,Count] = y - yref
    dispz[:,Count] = z - zref

    Count = Count + 1

# Force the last displacement to be zero
dispx[:,Count] = 0.0
dispy[:,Count] = 0.0
dispz[:,Count] = 0.0

# Time stamps
BPM  = 60
Time = np.linspace(0,60.0/BPM,NumT)

# First map global node id from one set of mesh (Initial) 
# to the other (Clip_9)
fname = './Initial-mesh-complete/mesh-complete.exterior.vtp'
coord, GNID, NNP = read_vtp(fname)
if NNP != NPoint:
    print("Number of points does not match!")
fname = './Final/'+"mesh-complete.exterior_"+format(0, '05d')+'.txt'
coordref = np.loadtxt(fname)

GlobalNodeID = np.zeros(NPoint,dtype=np.int) - 1
for ii in range(NPoint):
    for jj in range(NPoint):
        dist = np.linalg.norm(coordref[ii,:]-coord[jj,:])
        if dist < 1e-5:
            GlobalNodeID[ii] = GNID[jj]
            break 
if -1 in GlobalNodeID:
    print('Error')

# Map global node id to local id
G2L = np.zeros(np.max(GlobalNodeID)+1,dtype=np.int) - 1
for i in range(NPoint):
    G2L[GlobalNodeID[i]] = i

Dir = './Initial-mesh-complete/mesh-surfaces/'
write_motion(Dir,'Aorta')
write_motion(Dir,'Mitral')
write_motion(Dir,'LV')


# import matplotlib.pyplot as plt

# plt.plot(dispx)
# plt.plot(dispy)
# plt.plot(dispz)
# plt.show()
