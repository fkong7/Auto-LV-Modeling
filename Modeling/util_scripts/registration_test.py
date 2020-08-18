import numpy as np
import SimpleITK as sitk

a = np.zeros((10, 10))
b = np.zeros((10, 10))

a[2:5, 1:4] = 1
b[5:8, 3:6] = 1
#b[3:6, 3:6] = 1

im1 = sitk.GetImageFromArray(a)
im2 = sitk.GetImageFromArray(b)

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(im1)
elastixImageFilter.SetMovingImage(im2)
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.Execute()

params = elastixImageFilter.GetTransformParameterMap()
print(params[0]['TransformParameters'])
