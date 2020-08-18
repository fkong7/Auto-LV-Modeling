import os
import sys
import glob
import SimpleITK as sitk
import numpy as np
from mpi4py import MPI

import argparse


def fix_labels(fn, out_path):
    ids = [0, 205, 420, 500, 550, 600, 820, 850]


    im = sitk.ReadImage(fn)
    pyarr = sitk.GetArrayFromImage(im)
    for index, label in enumerate(ids):
        pyarr[pyarr==index]= label
    im_new = sitk.GetImageFromArray(pyarr)
    im_new.SetOrigin(im.GetOrigin())
    im_new.SetDirection(im.GetDirection())
    im_new.SetSpacing(im.GetSpacing())

    spacing = im.GetSpacing()
    kernel = [int(round(7./spacing[i])) for i in range(3)]
    kernel = [7 if kernel[i]>7 else kernel[i] for i in range(3)]
    ftr = sitk.BinaryMorphologicalClosingImageFilter()
    ftr.SetKernelRadius(kernel)
    ftr.SafeBorderOn()
    for i in ids:
        if i ==0:
            continue
        ftr.SetForegroundValue(int(i))
        im_new = ftr.Execute(im_new)

    sitk.WriteImage(im_new, os.path.join(out_path, os.path.basename(fn)))

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='Name of the folder containing segmentation to process')
    parser.add_argument('--output', help='Name of the output folder')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total = comm.Get_size()

    args = parser.parse_args()
    
    fns = sorted(glob.glob(os.path.join(args.folder, '*.nii.gz'))) 
    if rank ==0:
      print("number of segmentation data %d" % len(fns))
      try:
          os.makedirs(args.output)
      except Exception as e: print(e)

    num_vol_per_core = int(np.floor(len(fns)/total))
    extra = len(fns) % total
    vol_ids = list(range(rank*num_vol_per_core,(rank+1)*num_vol_per_core))
    if rank < extra:
        vol_ids.append(len(fns)-1-rank)

    seg_fn = [fns[k] for k in vol_ids]
    
    for n in seg_fn:
        fix_labels(n, args.output)
