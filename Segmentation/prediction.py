import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import gc
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import models as models_keras

import SimpleITK as sitk 
from preProcess import swapLabelsBack, resample_spacing, isometric_transform, centering, RescaleIntensity, swapLabels
from loss import bce_dice_loss, dice_loss
from tensorflow.python.keras import backend as K
from model import UNet2D
from imageLoader import ImageLoader
import argparse
import time
def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    print(matrix)
    return matrix 

def vtk_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    import vtk
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def appendPolyData(poly_list):
    """
    Combine two VTK PolyData objects together
    Args:
        poly_list: list of polydata
    Return:
        poly: combined PolyData
    """
    import vtk
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput() 
    return out
def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type

    vtkArray = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    #vtkArray = numpy_to_vtk(img.flatten())
    return vtkArray


def exportSitk2VTK(sitkIm,spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    import vtk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    print("Shape check: ", img.shape, sitkIm.GetSpacing())
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    print("Matrix check: ", matrix)
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    #imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix

def swapLabels_LH(labels):
    labels[labels==421]=420
    unique_label = np.unique(labels)

    new_label = range(len(unique_label))
    for i in range(len(unique_label)):
        label = unique_label[i]
        print(label)
        newl = new_label[i]
        print(newl)
        labels[labels==label] = newl
    
    if len(unique_label) != 4:
        labels[labels==1] = 0
        labels[labels==4] = 0
        labels[labels==5] = 0
        labels[labels==7] = 0
        labels[labels==2] = 1
        labels[labels==3] = 2
        labels[labels==6] = 3
       
    print(unique_label, np.unique(labels))

    return labels
def model_output_no_resize(model, im_vol, view, channel):
    im_vol = np.moveaxis(im_vol, view, 0)
    ipt = np.zeros([*im_vol.shape,channel])
    #shift array by channel num. If on boundary, fuse with
    #the slice on the other boundary
    shift = int((channel-1)/2)
    for i in range(channel):
        ipt[:,:,:,i] = np.roll(im_vol, shift-i, axis=0)
    start = time.time()
    prob = model.predict(ipt)
    end = time.time()
    prob = np.moveaxis(prob, 0, view)
    return prob, end-start

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

    def volume_prediction_average(self, size):
        self.input_size = size
        img_vol = resample_spacing(self.image_fn, order=1, template_size=(size, size, size) )[0]
        self.image_info = {}
        self.image_info['spacing'] = img_vol.GetSpacing()
        self.image_info['origin'] = img_vol.GetOrigin()
        self.image_info['direction'] = img_vol.GetDirection()

        img_vol = sitk.GetArrayFromImage(img_vol)


        img_vol = RescaleIntensity(img_vol,self.modality, [750, -750])
        
        
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
    
    def evaluate_dice(self):
        reference_segmentation = sitk.Cast(sitk.ReadImage(self.label_fn), sitk.sitkUInt16)
        self.pred = sitk.Cast(self.pred, sitk.sitkUInt8)
        ref_py = sitk.GetArrayFromImage(reference_segmentation)
        ref_py = swapLabels(ref_py)
        pred_py = sitk.GetArrayFromImage(self.pred)
        dice_values = dice_score(pred_py, ref_py)
        return dice_values
    def evaluate_assd(self):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        def _get_assd(p_surf, g_surf):
            dist_fltr = vtk.vtkDistancePolyDataFilter()
            dist_fltr.SetInputData(1, p_surf)
            dist_fltr.SetInputData(0, g_surf)
            dist_fltr.SignedDistanceOff()
            dist_fltr.Update()
            distance_poly = vtk_to_numpy(dist_fltr.GetOutput().GetPointData().GetArray(0))
            return np.mean(distance_poly), dist_fltr.GetOutput()
        ref_im =  sitk.ReadImage(self.label_fn)
        ref_im = resample_spacing(ref_im, template_size=(256,256,256), order=0)[0]
        ref_im, M = exportSitk2VTK(ref_im)
        pred_im = resample_spacing(self.pred, template_size=(256,256,256), order=0)[0]
        pred_im, M = exportSitk2VTK(pred_im)
        ref_im_py = swapLabels(vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        ref_im.GetPointData().SetScalars(numpy_to_vtk(ref_im_py))
        ids = np.unique(ref_im_py)
        pred_poly_list = []
        dist_poly_list = []
        ref_poly_list = []
        dist = [0.]*(len(ids)+1)
        for index, i in enumerate(ids):
            if i==0:
                continue
            p_s = vtk_marching_cube(pred_im, 0, i)
            r_s = vtk_marching_cube(ref_im, 0, i)
            dist_ref2pred, d_ref2pred = _get_assd(p_s, r_s)
            dist_pred2ref, d_pred2ref = _get_assd(r_s, p_s)
            dist[index+1] = (dist_ref2pred+dist_pred2ref)*0.5
            dist_poly_list.append(d_pred2ref)
            pred_poly_list.append(p_s)
            ref_poly_list.append(r_s)
        dist_poly = appendPolyData(dist_poly_list)
        pred_poly = appendPolyData(pred_poly_list)
        ref_poly = appendPolyData(ref_poly_list)
        dist[0], _ = _get_assd(pred_poly, ref_poly)
        return dist
    
    def resample_prediction(self, if_upsample=False):
        #resample prediction so it matches the original image
        print(self.pred.shape)
        im = sitk.GetImageFromArray(self.pred)
        im.SetSpacing(self.image_info['spacing'])
        im.SetOrigin(self.image_info['origin'])
        im.SetDirection(self.image_info['direction'])
        if if_upsample:
            ori_im = sitk.ReadImage(self.image_fn)
            size = ori_im.GetSize()
            spacing = ori_im.GetSpacing()
            new_size = [max(s,self.input_size) for s in size]
            ref_im = sitk.Image(new_size, ori_im.GetPixelIDValue())
            ref_im.SetOrigin(ori_im.GetOrigin())
            ref_im.SetDirection(ori_im.GetDirection())
            ref_im.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, size, spacing)])
            ctr_im = sitk.Resample(ori_im, ref_im, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear)    
            self.pred = centering(im, ctr_im, order=0)
        else:
            self.pred = centering(im, sitk.ReadImage(self.image_fn), order=0)
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
                #self.pred = ftr.Execute(ftr2.Execute(self.pred))
                #self.pred = self.pred
                self.pred = ftr.Execute(self.pred)
            else:
                self.pred = ftr.Execute(self.pred)


    def write_prediction(self, out_fn):
        try:
            os.makedirs(os.path.dirname(out_fn))
        except:
            pass
        sitk.WriteImage(sitk.Cast(self.pred, sitk.sitkInt16), out_fn)

def main(size, modality, data_folder, data_out_folder, model_folder, view_attributes, mode, channel, folder_postfix, upsample_prediction):
    print(modality)
    print(view_attributes)
    print(mode)
    print(os.path.join(data_out_folder, '%s_test.csv' % "ct"))

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
        im_loader = ImageLoader(m, data_folder, fn='_'+folder_postfix, fn_mask=None if mode=='test' else '_test_masks', ext='*.nii.gz')
        x_filenames, y_filenames = im_loader.load_imagefiles()
        im_loader = ImageLoader(m, data_folder, fn='_'+folder_postfix, fn_mask=None if mode=='test' else '_test_masks', ext='*.nii')
        x_filenames2, y_filenames2 = im_loader.load_imagefiles()
        x_filenames += x_filenames2
        y_filenames += y_filenames2
        dice_list = []
        assd_list = []
        for i in range(len(x_filenames)):
            print("processing "+x_filenames[i])
            models = [os.path.realpath(i) + '/weights_multi-all-%s_%s.hdf5' % (j, model_postfix) for i, j in zip(model_folders, view_names)]
            predict = Prediction(unet, models,m,view_attributes,x_filenames[i],y_filenames[i], channel)
            predict.volume_prediction_average(size)
            time_pred_list.append(predict.pred_time)
            predict.resample_prediction(upsample_prediction)
            #predict.post_process(m)

            predict.write_prediction(os.path.join(data_out_folder,os.path.basename(x_filenames[i])))

            time_list.append(time.time()-t_start)
            t_start = time.time()
            if y_filenames[i] is not None:
                dice_list.append(predict.evaluate_dice())
                assd_list.append(predict.evaluate_assd())
        if len(dice_list) >0:
            csv_path = os.path.join(data_out_folder, '%s_test.csv' % m)
            writeDiceScores(csv_path, dice_list)
            csv_path = os.path.join(data_out_folder, '%s_test_assd.csv' % m)
            writeDiceScores(csv_path, assd_list)
    return time_list, time_pred_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  help='Name of the folder containing the image data')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--model', nargs='+',  help='Name of the folders containing the trained models')
    parser.add_argument('--view', type=int, nargs='+', help='List of views for single or ensemble prediction, split by space. For example, 0 1 2  axial(0), coronal(1), sagittal(2)')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--size', type=int,default=256, help='Size of images')
    parser.add_argument('--mode', help='Test or validation (without or with ground truth label')
    parser.add_argument('--im_folder_postfix', default='test', help='Postfix of the folder containing image data, for ct_test, enter test')
    parser.add_argument('--n_channel',type=int, default=1, help='Number of image channels of input')
    parser.add_argument('--upsample_prediction', action='store_true', help='Keep the lowest dimension of the prediction to be the size of input')
    args = parser.parse_args()
    print('Finished parsing...')
    
    t_start = time.time() 
    time_list, time_pred_list = main(args.size, args.modality, args.image, args.output, args.model, args.view, args.mode, args.n_channel, args.im_folder_postfix, args.upsample_prediction)
    time_list.append(np.mean(time_list))
    #time_list.append(time.time()-t_start)
    time_pred_list.append(np.mean(time_pred_list))
    
    np.savetxt(os.path.join(args.output, 'time_results.csv'), np.transpose([np.array(time_list)]), fmt='%1.3f')
    np.savetxt(os.path.join(args.output, 'time_pred_results.csv'), np.transpose([np.array(time_pred_list)]), fmt='%1.3f')
