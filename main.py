import os
import numpy as np
from label_io import loadLabelMap
from marching_cube import marching_cube
from plot import plot_surface

fn = os.path.join(os.path.dirname(__file__), "examples", "ct_train_1002_label.nii.gz")

#load label map
label, _ = loadLabelMap(fn)
#convert to binary
label[np.where(label!=0)] = 1

#run marchine cube algorithm
output = marching_cube(label, 0)
verts, faces, _, _ = output

#plot
plot_surface(verts, faces)
