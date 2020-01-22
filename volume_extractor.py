import os
import glob
import vtk
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import label_io
import numpy as np
from scipy.interpolate import interp1d

def get_volume(poly):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetVolume()*1.e-3

if __name__ == '__main__':
    DIR_NAME = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Image_based/meshes_MACS40282_20150504/surfaces/'
    #DIR_NAME = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Image_based/meshes_MACS40244_20150309/surfaces/'
    START_PHASE = 8
    TOTAL_PHASE = 10

    fns = sorted(glob.glob(os.path.join(DIR_NAME, '*.vtk')))
    vols = []
    for i in list(range(START_PHASE, TOTAL_PHASE))+list(range(0, START_PHASE)):
        vol = get_volume(label_io.loadVTKMesh(fns[i]))
        vols.append(vol)

    vols +=vols
    vols.append(vols[0])
    x = np.linspace(0, 2, len(vols))
    f2 = interp1d(x, np.array(vols), kind='cubic')
    x2 = np.linspace(0, 1, 200)
    plt.plot(x2, f2(x2+1.), linewidth=2)
    x3 = np.linspace(0, 1, 11)

    #plt.plot(x3,  vols[:11])
    plt.xlabel('Time(s)')
    plt.ylabel('Volume (ml)')

    plt.show()
