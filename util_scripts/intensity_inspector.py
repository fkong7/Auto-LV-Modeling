import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from imageLoader import ImageLoader
from preProcess import RescaleIntensity

def mean_intensity(image, label,m,limit):
    """
    Output the mean intensity corresponding to each structure

    Args:
        image: SimpleITK image, input image
        label: SimpleITK image, input label
    Returns:
        values: np array of intensity values
    """
    py_im = RescaleIntensity(sitk.GetArrayFromImage(image), m, limit)
    py_label = sitk.GetArrayFromImage(label)
    py_label[py_label==421] =420
    assert py_label.shape == py_im.shape, "Image and label dimension do not match."
    labels = np.unique(py_label)

    values = np.zeros(0)
    for lb in labels:
        if lb ==0:
            continue
        mask = py_label == lb
        values = np.append(values, np.mean(py_im[mask]))
    print(values)
    return values

if __name__ == '__main__':

    im_dir = '/Users/fanweikong/Documents/ImageData/MMWHS'
    modality = ["ct", "mr"]

    intensity = {}
    for m in modality:
        im_loader = ImageLoader(m, im_dir, fn='_train', fn_mask='_train_masks')
        im_loader.load_imagefiles()
        values = []
        for x, y in zip(im_loader.x_filenames, im_loader.y_filenames):
            im = sitk.ReadImage(x)
            label = sitk.ReadImage(y)
            try:
                values = np.vstack((mean_intensity(im, label, m, [750, -750]), values))
            except Exception as e:
                print(e)
                values = mean_intensity(im, label, m, [750, -750])
        intensity[m] = values

    #plotting
    structure_names=['Myo', 'LV', 'LA', 'RA', 'RV', 'AA', 'PA']
    df_mr = pd.DataFrame(intensity["mr"], columns=structure_names)
    df_ct = pd.DataFrame(intensity["ct"], columns=structure_names)
    df_mr['Key'] = "mr"
    df_ct['Key'] = "ct"
    print(df_mr)
    print(df_ct)
    
    df = pd.concat([df_ct,df_mr],keys=['ct','mr'])
    group = df.groupby(['Key'])
    boxplot = group.boxplot()
    plt.show()
    
