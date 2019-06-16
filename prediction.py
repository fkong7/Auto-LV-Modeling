import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import models as models_keras

import SimpleITK as sitk 
from skimage.transform import resize
from preProcess import swapLabelsBack
from loss import bce_dice_loss, dice_loss
from tensorflow.python.keras import backend as K
from model import UNet2D
from imageLoader import ImageLoader

def model_output_no_resize(model, im_vol, view, modality):
    im_vol = np.moveaxis(im_vol, view, 0)
    prob = model.predict(np.expand_dims(im_vol, axis=-1))
    prob = np.moveaxis(prob, 0, view)
    return prob

def predictVol(prob,labels):
    #im_vol, ori_shape, info = data_preprocess_test(image_vol_fn, view, 256, modality)
    predicted_label = np.argmax(prob, axis=-1)

    predicted_label = swapLabelsBack(labels,predicted_label)
    return predicted_label

from scipy.spatial.distance import dice
def dice_score(pred, true):
  pred = pred.astype(np.int)
  true = true.astype(np.int)  
  num_class = np.unique(true)
  dice_out = [None]*len(num_class)
  
  for i in range(len(num_class)):
    pred_i = pred==num_class[i]
    true_i = true==num_class[i]
    sim = 1 - dice(pred_i.reshape(-1), true_i.reshape(-1))
    dice_out[i] = sim
    
  return dice_out


import csv
def writeDiceScores(csv_path,dice_outs): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow( ('Bg 0', 'myo 205', 'la 420', 'lv 500', 'ra 550', 'rv 600', 'aa 820', 'pa 850') )
        for i in range(len(dice_outs)):
            writer.writerow(tuple(dice_outs[i]))
            print(dice_outs[i])
  
    writeFile.close()


class Prediction:
    #This is a class to get 3D volumetric prediction from the 2DUNet model
    def __init__(self, unet, model,modality,view,image_vol,label_vol):
        self.unet=unet
        self.models=model
        self.modality=modality
        self.views=view
        self.image_vol = image_vol
        self.label_vol = label_vol
        self.prediction = None
        self.dice_score = None
        self.original_shape = image_vol.GetSize()
        self.image_info = (image_vol.GetOrigin(), image_vol.GetSpacing(), image_vol.GetDirection())
        assert len(self.models)==len(self.views), "Missing view attributes for models"

    def volume_prediction_average(self, size):
        img_vol = sitk.GetArrayFromImage(self.image_vol)
        label_vol = sitk.GetArrayFromImage(self.label_vol)

        prob = np.zeros((*self.original_shape,8))
        unique_views = np.unique(self.views)
        
        for view in unique_views:
            indices = np.where(self.views==view)[0]
            predict_shape = [size,size,size,8]
            predict_shape[view] = img_vol.shape[view]
            prob_view = np.zeros(predict_shape)
            for i in indices:
                model_path = self.models[i]
                image_vol_resize = Resize_by_view(img_vol, self.views[i], size)
                (self.unet).load_weights(model_path)
                prob_view+=model_output_no_resize(self.unet, image_vol_resize, self.views[i], self.modality)
            prob_resize = np.zeros(prob.shape)
            for i in range(prob.shape[-1]):
                prob_resize[:,:,:,i] = resize(prob_view[:,:,:,i], self.original_shape, order=1)
            prob += prob_resize
        avg = prob/len(self.models)
        if out_fn:
            np.save(out_fn, avg)
        self.prediction = predictVol(avg, label_vol)
        return self.prediction

    def dice(self):
        label_vol = sitk.GetArrayFromImage(self.label_vol)
        self.dice_score = dice_score(self.prediction, self.label_vol)
        return self.dice_score
    
    def resample_prediction(self):
        #resample prediction so it matches the original image
        transformed = isometric_transform(sitk.GetImageFromArray(self.prediction), self.label_vol,np.eye(3),order=0,target=self.label_vol.GetDirection())
        transformed = centering(transformed, self.label_vol, order=0)
        self.prediction = sitk.GetArrayFromImage(transformed)
        return self.prediction

    def write_prediction(self, out_fn):
        ori, space, direc = self.image_info
        out_im = sitk.GetImageFromArray(self.prediction)
        out_im.SetOrigin(ori)
        out_im.SetSpacing(space)
        out_im.SetDirection(direc)
      
        sitk.WriteImage(sitk.Cast(out_im, sitk.sitkInt16), out_fn)

def main():
    modality = ["mr", "ct"]
    im_base_folder = "MMWHS_small_aug"
    data_folder = '/global/scratch/fanwei_kong/DeepLearning/' + im_base_folder 
    folder_postfix = "aug_both"
    model_postfix = "small2"
    base_folder = ["MMWHS_small_aug","MMWHS_small_aug"]
    names = ['axial', 'coronal', 'sagittal']
    view_attributes = [0]
    view_names = [names[i] for i in view_attributes]
    data_out_folder = '/global/scratch/fanwei_kong/DeepLearning/2DUNet/Logs/%s/prediction_%s' % (base_folder[-1], folder_postfix)
    try:
      os.mkdir('/global/scratch/fanwei_kong/DeepLearning/2DUNet/Logs/%s' % base_folder[-1])
      os.mkdir(data_out_folder)
    except Exception as e: print(e)
    
    #set up models
    img_shape = (256, 256, 1)
    num_class = 8
    inputs, outputs = UNet2D(img_shape, num_class)
    unet = models_keras.Model(inputs=[inputs], outputs=[outputs])
    
    #load image filenames
    filenames = {}
    for m in modality:
        im_loader = ImageLoader(m, data_folder, fn='_test', fn_mask='_test_masks')
        x_filenames, y_filenames = im_loader.load_imagefiles()
        dice_list = [None]*len(x_filenames)

        for i in range(len(x_filenames)):
            print("processing "+x_filenames[i])
            models = ['/global/scratch/fanwei_kong/DeepLearning/2DUNet/Logs/%s/weights_multi-all-%s_%s.hdf5' % (base_folder[j], view_names[j], model_postfix) for j in range(len(view_attributes))]
            img, _ = resample_spacing(x_filenames[i], order=1)
            mask = sitk.ReadImage(y_filenames[i])
            predict = Prediction(unet, models,m,view_attributes,img,mask)
            predict.volume_prediction_average(256)
            predict.resample_prediction()
            dice_list[i] = predict.dice()
            
            predict.write_prediction(os.path.join(data_out_folder,os.path.basename(x_filenames[i])))
            del predict 
        csv_path = '/global/scratch/fanwei_kong/2DUNet/Logs/%s/%s_test-%s.csv' % (base_folder[-1], m , folder_postfix) 
        writeDiceScores(csv_path, dice_list)
    