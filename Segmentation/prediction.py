import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import gc
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import models as models_keras

import SimpleITK as sitk 
import vtk
from preProcess import swapLabelsBack, resample_spacing, isometric_transform, centering, RescaleIntensity, vtk_resample_to_size, vtk_reslice_image, vtk_resample_with_info_dict
from utils import load_vtk_image, writeVTKImage, get_array_from_vtkImage,get_vtkImage_from_array,vtk_write_mask_as_nifty
from loss import bce_dice_loss, dice_loss
from tensorflow.python.keras import backend as K
from model import UNet2D
from imageLoader import ImageLoader
import argparse
import time

def model_output_no_resize(model, im_vol, view, channel):
    im_vol = np.moveaxis(im_vol, view, 0)
    ipt = np.zeros([*im_vol.shape,channel])
    #shift array by channel num. If on boundary, fuse with
    #the slice on the other boundary
    shift = int((channel-1)/2)
    for i in range(channel):
        ipt[:,:,:,i] = np.roll(im_vol, shift-i, axis=0)
    start = time.time()
    prob = np.zeros(list(ipt.shape[:-1])+[model.layers[-1].output_shape[-1]])
    for i in range(prob.shape[0]):
        slice_im = np.expand_dims(ipt[i,:,:,:],axis=0)
        prob[i,:,:,:] = model.predict(slice_im)
    end = time.time()
    prob = np.moveaxis(prob, 0, view)
    return prob, end-start

def predictVol(prob,labels):
    #im_vol, ori_shape, info = data_preprocess_test(image_vol_fn, view, 256, modality)
    predicted_label = np.argmax(prob, axis=-1)

    predicted_label = swapLabelsBack(labels,predicted_label)
    return predicted_label

def dice_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)  
    num_class = np.unique(true)
    
    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))
    
    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out


import csv
def writeDiceScores(csv_path,dice_outs): 
    with open(csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow( ('Total', 'myo 205', 'la 420', 'lv 500', 'ra 550', 'rv 600', 'aa 820', 'pa 850') )
        for i in range(len(dice_outs)):
            writer.writerow(tuple(dice_outs[i]))
            print(dice_outs[i])
  
    writeFile.close()


class Prediction:
    #This is a class to get 3D volumetric prediction from the 2DUNet model
    def __init__(self, unet, model,modality,view,image_fn,label_fn, channel):
        self.unet=unet
        self.models=model
        self.modality=modality
        self.views=view
        self.image_fn = image_fn
        self.channel = channel
        self.label_fn = label_fn
        self.prediction = None
        self.dice_score = None
        self.original_shape = None
        assert len(self.models)==len(self.views), "Missing view attributes for models"
    def prepare_input_sitk(self, size):
        img_vol = resample_spacing(self.image_fn, order=1, template_size=(size, size, size) )[0]
        self.image_info = {}
        self.image_info['spacing'] = img_vol.GetSpacing()
        self.image_info['origin'] = img_vol.GetOrigin()
        self.image_info['direction'] = img_vol.GetDirection()
        img_vol = sitk.GetArrayFromImage(img_vol)
        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        #img_vol = (img_vol - np.mean(img_vol))/np.std(img_vol)
        return img_vol 
    def prepare_input_vtk(self, size):
        vtk_img = load_vtk_image(self.image_fn)
        self.image_info = {}
        self.image_info['spacing'] = vtk_img.GetSpacing()
        self.image_info['origin'] = vtk_img.GetOrigin()
        self.image_info['extent'] = vtk_img.GetExtent()
        self.image_info['size'] = vtk_img.GetDimensions()
        vtk_img = vtk_resample_to_size(vtk_img, (size, size, size))
        img_vol = get_array_from_vtkImage(vtk_img)
        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        return img_vol
    def volume_prediction_average(self, size):

        img_vol = self.prepare_input_sitk(size)
        
        self.original_shape = img_vol.shape
        
        prob = np.zeros((*self.original_shape,8))
        unique_views = np.unique(self.views)
        
        self.pred_time = 0.
        for view in unique_views:
            indices = np.where(self.views==view)[0]
            predict_shape = [size,size,size,8]
            predict_shape[view] = img_vol.shape[view]
            prob_view = np.zeros(predict_shape)
            for i in indices:
                model_path = self.models[i]
                (self.unet).load_weights(model_path)
                p, t = model_output_no_resize(self.unet, img_vol, self.views[i], self.channel)
                prob_view += p
                self.pred_time += t
            prob += prob_view
        avg = prob/len(self.models)
        self.pred = predictVol(avg, np.zeros(1))
        return 

    def dice(self):
        #assuming groud truth label has the same origin, spacing and orientation as input image
        label_vol = sitk.GetArrayFromImage(sitk.ReadImage(self.label_fn))
        pred_label = sitk.GetArrayFromImage(self.pred)
        pred_label = swapLabelsBack(label_vol, pred_label)
        ds = dice_score(pred_label, label_vol)
        return ds
    
    def resample_prediction(self):
        #resample prediction so it matches the original image
        im = sitk.GetImageFromArray(self.pred)
        im.SetSpacing(self.image_info['spacing'])
        im.SetOrigin(self.image_info['origin'])
        im.SetDirection(self.image_info['direction'])
        self.pred = centering(im, sitk.ReadImage(self.image_fn), order=0)
        return
    def resample_prediction_vtk(self):
        im = get_vtkImage_from_array(self.pred.astype(np.uint8))
        self.pred = vtk_resample_with_info_dict(im, self.image_info, order=0)
        return 
    def post_process(self, m):
        spacing = self.pred.GetSpacing()
        ids = np.unique(sitk.GetArrayFromImage(self.pred))
        kernel = [int(round(5./spacing[i])) for i in range(3)]
        #Max kernel size is 7
        kernel = [5 if kernel[i]>5 else kernel[i] for i in range(3)]
        ftr = sitk.BinaryMorphologicalClosingImageFilter()
        ftr.SetKernelRadius(kernel)
        ftr.SafeBorderOn()
        ftr2 = sitk.BinaryMorphologicalOpeningImageFilter()
        ftr2.SetKernelRadius([2, 2, 2])
        for i in ids:
            if i ==0:
                continue
            ftr.SetForegroundValue(int(i))
            ftr2.SetForegroundValue(int(i))
            if m =="ct":
                self.pred = ftr.Execute(self.pred)
            else:
                self.pred = ftr.Execute(self.pred)

    def write_prediction(self, out_fn):
        try:
            os.makedirs(os.path.dirname(out_fn))
        except Exception as e: print(e)
        sitk.WriteImage(sitk.Cast(self.pred, sitk.sitkInt16), out_fn)
        return 


def main(size, modality, data_folder, data_out_folder, model_folder, view_attributes, mode, channel, folder_postfix):

    time_list = []
    time_pred_list = []

    model_postfix = "small2"
    model_folders = sorted(model_folder * len(view_attributes))
    view_attributes *= len(model_folder)

    names = ['axial', 'coronal', 'sagittal']
    view_names = [names[i] for i in view_attributes]
    try:
      os.mkdir(data_out_folder)
    except Exception as e: print(e)
    
    #set up models
    img_shape = (size, size, channel)
    num_class = 8
    inputs, outputs = UNet2D(img_shape, num_class)
    unet = models_keras.Model(inputs=[inputs], outputs=[outputs])
    
    #load image filenames
    filenames = {}

    t_start = time.time()
    for m in modality:
        im_loader = ImageLoader(m, data_folder, fn='_'+folder_postfix, fn_mask=None if mode=='test' else '_test_seg', ext='*.nii.gz')
        x_filenames, y_filenames = im_loader.load_imagefiles()
        im_loader = ImageLoader(m, data_folder, fn='_'+folder_postfix, fn_mask=None if mode=='test' else '_test_seg', ext='*.nii')
        x_filenames2, y_filenames2 = im_loader.load_imagefiles()
        x_filenames += x_filenames2
        y_filenames += y_filenames2
        dice_list = []
        for i in range(len(x_filenames)):
            print("Processing "+x_filenames[i])
            models = [os.path.realpath(i) + '/weights_multi-all-%s_%s.hdf5' % (j, model_postfix) for i, j in zip(model_folders, view_names)]
            predict = Prediction(unet, models,m,view_attributes,x_filenames[i],y_filenames[i], channel)
            predict.volume_prediction_average(size)
            time_pred_list.append(predict.pred_time)
            predict.resample_prediction()
            #predict.post_process(m)

            predict.write_prediction(os.path.join(data_out_folder,os.path.basename(x_filenames[i])))

            time_list.append(time.time()-t_start)
            t_start = time.time()
            if y_filenames[i] is not None:
                dice_list.append(predict.dice())
        if len(dice_list) >0:
            csv_path = os.path.join(data_out_folder, '%s_test.csv' % m)
            writeDiceScores(csv_path, dice_list)
    return time_list, time_pred_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  help='Name of the folder containing the image data')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--model', nargs='+',  help='Name of the folders containing the trained models')
    parser.add_argument('--view', type=int, nargs='+', help='List of views for single or ensemble prediction, split by space. For example, 0 1 2  axial(0), coronal(1), sagittal(2)')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--size', type=int,default=256, help='Size of images')
    parser.add_argument('--mode', default='test',help='Test or validation (without or with ground truth label')
    parser.add_argument('--im_folder_postfix', default='test', help='Postfix of the folder containing image data, for ct_test, enter test')
    parser.add_argument('--n_channel',type=int, default=1, help='Number of image channels of input')
    args = parser.parse_args()
    
    t_start = time.time() 
    time_list, time_pred_list = main(args.size, args.modality, args.image, args.output, args.model, args.view, args.mode, args.n_channel, args.im_folder_postfix)
    time_list.append(np.mean(time_list))
    #time_list.append(time.time()-t_start)
    time_pred_list.append(np.mean(time_pred_list))
    
    np.savetxt(os.path.join(args.output, 'time_results.csv'), np.transpose([np.array(time_list)]), fmt='%1.3f')
    np.savetxt(os.path.join(args.output, 'time_pred_results.csv'), np.transpose([np.array(time_pred_list)]), fmt='%1.3f')
