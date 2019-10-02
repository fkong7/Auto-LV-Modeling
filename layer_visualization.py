"""
Scripts for visualizing the intermediate layer output from trained models"
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import offsetbox
from skimage.transform import resize
from sklearn import manifold


import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import SimpleITK as sitk 

from utils import getTrainNLabelNames
from preProcess import swapLabelsBack, resample_spacing, Resize_by_view, isometric_transform, centering, RescaleIntensity
    
from loss import bce_dice_loss, dice_loss
from model import UNet2D

from mpi4py import MPI

def getLayerFuncByName(model, layername):
    """
    Get function pointer for layer outputs from input

    Args:
        model: keras model
        layername: str, name of layer
    Returns:
        get_layer_outputs: function pointer
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            get_layer_outputs = K.function([model.layers[0].input],
                                                      [model.layers[idx].output])
    return get_layer_outputs

def plot_embedding(X, y, title=None,classes=["ct", "mr"]):
    """
    Plots the t-sne embedding computed

    Args:
        X: numpy array after t-sne embedding
        Y: class labels for each sample
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(np.unique(y))):
        X_c = X[y==i,:]
        plt.scatter(X_c[:,0], X_c[:,1],color=plt.cm.tab10(i/ 10.), label=classes[i] )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
            # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def preProcessImage(img_fn, mask_fn, m, view, size=None):
    from skimage.transform import resize
    """
    pre-process image and labels to feed into models 
    """
    img, _ = resample_spacing(img_fn, order=1)
    mask, _ = resample_spacing(mask_fn, order=0)
                
    img_vol = sitk.GetArrayFromImage(img)
    label_vol = sitk.GetArrayFromImage(mask)
    
    img_vol = RescaleIntensity(img_vol, m, [750, -750])

    #remove blank slices
    label_vol = np.moveaxis(label_vol, view, 0)
    IDs = np.max(np.max(label_vol,axis=-1),axis=-1)==0
    sliced = np.moveaxis(img_vol,view,0)[~IDs,:,:]
    if size is not None: #rescale the view axis
        shape = (size, 256, 256)
    else:
        shape = (sliced.shape[0], 256, 256)
    sliced = np.moveaxis(resize(sliced, shape, order=1), 0, view)
    mask_sliced = label_vol[~IDs,:,:]
    mask_sliced = np.moveaxis(resize(mask_sliced, shape, order=1), 0, view)
                      
    return sliced, mask_sliced


def main():
    save_model_path = '/global/scratch/fanwei_kong/DeepLearning/2DUNet/Logs/MMWHS_small_aug/ct_mr_combined/weights_multi-all-axial_small2.hdf5'
    data_folder =     '/global/scratch/fanwei_kong/DeepLearning/ImageData/MMWHS_small_aug'
    layername = 'conv2d_22'
    save_figname = os.path.join(os.path.dirname(save_model_path), "layer_"+layername+'.png')
    np_fn_layer = os.path.join(os.path.dirname(save_model_path), "layer_"+layername+'_layer.npy')
    np_fn_label = os.path.join(os.path.dirname(save_model_path), "layer_"+layername+'_label.npy')
    view = 0
    num_slices = 40
    trial_num = 24 # for debug
    #save_model_path = '/content/gdrive/My Drive/DeepLearning/2DUnet-git/Logs/MMWHS_small_aug/ct_only/weights_multi-all-axial_small2.hdf5'
    #save_model_path = '/content/gdrive/My Drive/DeepLearning/2DUnet-git/Logs/MMWHS_small_aug/mr_only/weights_multi-all-axial_small2.hdf5'
    
    #Load model
    img_shape = (256, 256, 1)
    inputs, outputs = UNet2D(img_shape, 8)
    unet = models.Model(inputs=[inputs], outputs=[outputs])
    unet.load_weights(save_model_path)
    #modality = ["ct", "mr"]
    modality = ["mr", "ct"]

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total = comm.Get_size()

    #load filenames
    filenames = {}
    if rank ==0:
        split_sizes = np.zeros(total) #initialize to sum up with all modality
    for m in modality:
        imgVol_fn, mask_fn = getTrainNLabelNames(data_folder, m)
        if trial_num > 0: # DEBUG
            imgVol_fn = imgVol_fn[:trial_num]
            mask_fn = mask_fn[:trial_num]
        # MPI 
        num_vol_per_core = int(np.floor(len(imgVol_fn)/comm.Get_size()))
        extra = len(imgVol_fn) % comm.Get_size()
        vol_ids = list(range(rank*num_vol_per_core,(rank+1)*num_vol_per_core))
        if rank < extra:
            vol_ids.append(len(imgVol_fn)-1-rank)
        
        if rank==0:
            print("Number of image and mask volumes for modality %s: %d, %d" % (m, len(imgVol_fn), len(mask_fn)))
            split_sizes += np.ones(total)*num_vol_per_core
            for i in range(extra):
                split_sizes[i]+=1

        filenames["x_"+m] = [imgVol_fn[k] for k in vol_ids]
        filenames["y_"+m] = [mask_fn[k] for k in vol_ids]
        print("RANK %d, numebr of image and mask volumes for modality %s: %d, %d" % (rank, m, len(filenames["x_"+m]), len(filenames["y_"+m])))
    
    get_layer_output = getLayerFuncByName(unet, layername)


    #start layer outputs
    outputs = {}
    for m in modality:
        outputs[m] = []
        for i in range(len(filenames["x_"+m])):
            print(filenames["x_"+m][i], filenames["y_"+m][i])
            features,_ = preProcessImage(filenames["x_"+m][i], filenames["y_"+m][i], modality[0], view, num_slices)
            layer_output = get_layer_output([np.expand_dims(features, axis=-1)])
            # down-size to fit within memory
            # outputs[m].append(resize(layer_output[0], tuple(int(x/2) for x in layer_output[0].shape)).flatten())
            outputs[m].append(layer_output[0].flatten())
        outputs[m] = np.asarray(outputs[m])


    # MPI GATHTER
    comm.Barrier()
    output_data = {}
    send_buff = {}
    send_buff['layer'] = np.concatenate(tuple(outputs[m] for m in modality))
    send_buff['label'] = np.concatenate([np.ones(outputs[modality[i]].shape[0])*i for i in range(len(modality))])
    print(send_buff['layer'])
    print(np.max(send_buff['layer']))
    if rank==0:
        print("Start gathering")
        split_sizes_output = split_sizes * np.prod(send_buff['layer'].shape[1:])
        print("split_sizes_output: ", split_sizes_output)
        displacement_output = {}
        displacement_output['layer'] = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        displacement_output['label'] = np.insert(np.cumsum(split_sizes), 0, 0)[0:-1]
        output_data['layer'] = np.zeros([int(np.sum(split_sizes)), int(np.prod(send_buff['layer'].shape[1:]))])
        output_data['label'] = np.zeros(int(np.sum(split_sizes)))
    else:
        split_sizes_output = None
        displacement_output = None
        split_sizes = None
        output_data['layer'] = None
        output_data['label'] = None
        
    split_sizes = comm.bcast(split_sizes, root=0)
    split_sizes_output = comm.bcast(split_sizes_output, root=0)
    displacement_output = comm.bcast(displacement_output, root=0)
    print(displacement_output)
    comm.Barrier()
    comm.Gatherv(send_buff['layer'],[output_data['layer'],split_sizes_output,displacement_output['layer'],MPI.DOUBLE], root=0) #Gather output data together
    comm.Gatherv(send_buff['label'],[output_data['label'],split_sizes,displacement_output['label'],MPI.DOUBLE], root=0) #Gather output data together
    
    comm.Barrier()
    if rank==0:
        print(output_data['layer'])
        print(np.max(output_data['layer']))
        np.save(np_fn_layer, output_data['layer'])
        np.save(np_fn_label, output_data['label'])
        # t-SNE embedding of the digits dataset
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(output_data['layer'])
        plot_embedding(X_tsne, output_data['label'],
                       "t-SNE embedding of the feature maps",
                       classes=modality)
        plt.savefig(save_figname, bbox_inches = "tight")
if __name__ =="__main__":
    main()
