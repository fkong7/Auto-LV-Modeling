import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import numpy as np
import SimpleITK as sitk
#import matplotlib.pyplot as plt
from imageLoader import ImageLoader
import preProcess
from preProcess import RescaleIntensity
def pre_process(image, label, m, limit):

    py_im = RescaleIntensity(sitk.GetArrayFromImage(image), m, limit)
    py_label = sitk.GetArrayFromImage(label)
    py_label[py_label==421] =420
    return py_im, py_label

def mean_intensity(py_im, py_label):
    """
    Output the mean intensity corresponding to each structure

    Args:
        image: numpy image, input image
        label: numpy image, input label
    Returns:
        values: np array of intensity values
    """
    assert py_label.shape == py_im.shape, "Image and label dimension do not match."
    labels = np.unique(py_label)

    values = np.zeros(0)
    for lb in labels:
        if lb ==0:
            continue
        mask = py_label == lb
        values = np.append(values, np.mean(py_im[mask]))
    return values

def intensity_plots():
    import pandas as pd
    im_dir = '/Users/fanweikong/Documents/ImageData/MMWHS_2'
    modality = ["ct", "mr", "ct", "mr", "mr", "mr"]
    folders = ['_train', '_train', '_val', '_val', '_special_test', '_special_test2']
    keys = ["ct", "mr", "ct_val", "mr_val", "mr_test_good", "mr_test_bad"]
    intensity = {}
    def get_mean_intensity(folder_names, m, key, intensity):
        im_loader = ImageLoader(m, im_dir, fn=folder_names[0], fn_mask=folder_names[1])
        im_loader.load_imagefiles()
        values = np.zeros((0,7))
        for x, y in zip(im_loader.x_filenames, im_loader.y_filenames):
            im = sitk.ReadImage(x)
            label = sitk.ReadImage(y)
            im, label = pre_process(im, label, m, [750, -750])
            #try histogram equalization
#            im = preProcess.HistogramEqualization(im)
            #im, label = apply_intensity_map(im , label)
            class_intensity = mean_intensity(im, label)
            print(x)
            print("Max, min: ", np.max(class_intensity), np.min(class_intensity))
            values = np.vstack((values, class_intensity))
        intensity[key] = values
        return intensity
    for m, key,  folder in zip(modality, keys, folders):
        intensity = get_mean_intensity([folder, folder+'_masks'], m, key, intensity)
    #plotting
    structure_names=['Myo', 'LV', 'LA', 'RA', 'RV', 'AA', 'PA']
    dfs = [None] * len(keys)
    for i, k in enumerate(keys):
        dfs[i] = pd.DataFrame(intensity[k], columns=structure_names)
        dfs[i]['Key'] = k
    
    df = pd.concat([d for d in dfs],keys=keys)

    fig, axes = plt.subplots(1,len(keys))
    group = df.groupby(['Key'], sort=False)
    for m, ax in zip(keys,axes):
        print(intensity[m])
        for i in range(intensity[m].shape[0]):
            y = intensity[m][i,:]
            ax.plot(np.array(range(1,len(structure_names)+1)), y, marker='o')
            ax.set_ylim([-1,1])
    boxplot = group.boxplot(ax=axes)
    plt.show()

def apply_intensity_map(tr_img, label_img):
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter
    # compute the mean intensity of each class, perturb them and compute a smooth intensity mapping
    
    means =  mean_intensity(tr_img, label_img)
    rng = np.max(tr_img) - np.min(tr_img)
    perturbed = np.clip(np.random.normal(means, 0.15*rng*0.5*np.ones(len(means))), np.min(tr_img), np.max(tr_img))
    print(means)
    print(perturbed)
    # add min and max to map the full range
    means = np.append(np.insert(means, 0, np.min(tr_img)), np.max(tr_img))
    perturbed = np.append(np.insert(perturbed, 0, np.min(tr_img)), np.max(tr_img))
    f = interpolate.interp1d(means, perturbed, kind='slinear')
    out_img = gaussian_filter(f(tr_img), sigma=1)
    #plt.figure()
    #x = np.linspace(-1,1,20)
    #y = f(x)
    #plt.plot(x, y)
    #plt.show()
    #fig, axes = plt.subplots(2,3)
    #for i in range(axes.shape[0]):
    #    loc = 50
    #    incr = 40
    #    axes[i][0].imshow(tr_img[loc+incr*(i+1),:,:], cmap='gray')
    #    axes[i][0].axis('off')
    #    axes[i][1].imshow(out_img[loc+incr*(i+1),:,:],cmap='gray')
    #    axes[i][1].axis('off')
    #    axes[i][2].imshow(label_img[loc+incr*(i+1),:,:],cmap='gray')
    #    axes[i][2].axis('off')
    #
    #plt.show()
    return out_img, label_img

def tf_intensity_augmentation(tr_img, label_img, num_class, changeIntensity):
    import tensorflow as tf 
    lb = [0, 550, 600, 205, 420, 500, 820, 850]
    if changeIntensity:
        for i in range(1,num_class):
            scale = tf.random_uniform([], 0.2, 0.3)
            pick = tf.random_uniform([], -0.9, 0.9)
            scaled = (pick - tr_img)*scale + tr_img
            #scaled = tr_img * scale
            tr_img = tf.where(tf.equal(label_img,lb[i]), scaled, tr_img)
    return tr_img, label_img

def tf_test(img_fn, label_fn):
    import tensorflow as tf 
    img = RescaleIntensity(sitk.GetArrayFromImage(preProcess.resample_spacing(img_fn)[0]), "mr", [750, -750])
    label = sitk.GetArrayFromImage(preProcess.resample_spacing(label_fn, order=0)[0])
    label[label==421] = 420    
    tf_img = tf.placeholder(tf.float32, shape=img.shape)
    tf_label = tf.placeholder(tf.int32, shape=label.shape)
    tr_img_aug, tr_label_aug= tf_intensity_augmentation(tf_img, tf_label, 8, changeIntensity=True)
    with tf.Session() as sess:
        out_im, out_label = sess.run([tr_img_aug, tr_label_aug], feed_dict={tf_img: img, tf_label: label})
        fig, axes = plt.subplots(2,3)
        for i in range(axes.shape[0]):
            loc = 50
            incr = 40
            axes[i][0].imshow(img[loc+incr*(i+1),:,:], cmap='gray')
            axes[i][0].axis('off')
            axes[i][1].imshow(out_im[loc+incr*(i+1),:,:],cmap='gray')
            axes[i][1].axis('off')
            axes[i][2].imshow(out_label[loc+incr*(i+1),:,:],cmap='gray')
            axes[i][2].axis('off')
        
        plt.show()
    return out_im, out_label
if __name__ == '__main__':
    #tf_test()
    intensity_plots()
