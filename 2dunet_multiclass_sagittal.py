# -*- coding: utf-8 -*-
"""2dUNet_multiclass.ipynb
"""
import os
import glob
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import h5py
import SimpleITK as sitk

from utils import getTrainNLabelNames
from preProcess import swapLabels
from preProcess import RescaleIntensity
from utils import np_to_tfrecords
from preProcess import data_preprocess

from sampling import sample
from sampling import bootstrapping

from augmentation import shift_img
from augmentation import flip_img
from augmentation import changeIntensity_img
from augmentation import _augment

from tf_dataset import _parse_function
from tf_dataset import _process_pathnames
from tf_dataset import get_baseline_dataset
from model import UNet2D

from loss import dice_coeff
from loss import dice_loss
from loss import bce_dice_loss

from tensorflow.python.keras.optimizers import Adam

from pickle import dump
import sys
"""# Set up"""

img_shape = (256, 256, 1)
batch_size = 10
#epochs = 100

#modality = ["ct","mr"]
modality = ["ct"]
#im_base_name = 'MMWHS_small_13'
#base_name = 'MMWHS_small_13'
im_base_name = sys.argv[1]
base_name = sys.argv[2]
seed = int(sys.argv[3])
num_class = int(sys.argv[4])
epochs = int(sys.argv[5])

data_folder = '/global/scratch/fanwei_kong/ImageData/%s' % im_base_name
view = 2
view_names = ['axial', 'coronal', 'sagittal']
data_folder_out = '/global/scratch/fanwei_kong/ImageData/%s/2d_multiclass-%s2_train' % (im_base_name,view_names[view])
data_val_folder_out = '/global/scratch/fanwei_kong/ImageData/%s/2d_multiclass-%s2_val' % (im_base_name,view_names[view])
save_model_path = '/global/scratch/fanwei_kong/2DUNet/Logs/%s/weights_multi-all-%s_small2.hdf5' % (base_name,view_names[view])
save_loss_path = '/global/scratch/fanwei_kong/2DUNet/Logs/%s/%s' % (base_name,view_names[view])

""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(save_loss_path))
except Exception as e: print(e)

for m in modality:
  try:
    os.mkdir(os.path.join(data_folder_out, m+'_train'))
    #os.mkdir(os.path.join(data_folder_out, m+'_train_masks'))
  except Exception as e: print(e)

""" Pre-process data """
def buildImageDataset(data_folder_out, modality, seed):
    x_train_filenames = []
    filenames = [None]*len(modality)
    nums = np.zeros(len(modality))
    for i, m in enumerate(modality):
      filenames[i], _ = getTrainNLabelNames(data_folder_out, m, ext='*.tfrecords')
      nums[i] = len(filenames[i])
      x_train_filenames+=filenames[i]
      
    print("Number of images obtained for training and validation: " + str(nums))
    
    """ Sample dataset """
    np.random.seed(seed)
    nums = np.max(nums) - nums
    for i , _ in enumerate(modality):
      index = sample(list(range(len(filenames[i]))), nums[i])
      x_train_filenames+=[filenames[i][j] for j in index]
    
    x_train_filenames = bootstrapping(x_train_filenames)

    return x_train_filenames

x_train_filenames = buildImageDataset(data_folder_out, modality, seed)
x_val_filenames = buildImageDataset(data_val_folder_out, modality, seed)
num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
print("Number of training examples after sampling: {}".format(num_train_examples))
print("Number of validation examples after sampling: {}".format(num_val_examples))





"""## Set up train and validation datasets
Note that we apply image augmentation to our training dataset but not our validation dataset.
"""

tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'horizontal_flip': True,
    'changeIntensity': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

train_ds = get_baseline_dataset(x_train_filenames, preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)
val_ds = get_baseline_dataset(x_train_filenames, preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)

"""# DEBUG """
print("Debug...")
data_aug_iter = train_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()
with tf.Session() as sess: 
    batch_of_imgs, label = sess.run(next_element)
    print("****DEBUG PRINT****")
    print(batch_of_imgs.shape)
    print(label.shape)


"""# Build the model"""
inputs, outputs = UNet2D(img_shape, num_class)

model = models.Model(inputs=[inputs], outputs=[outputs])

adam = Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dice_loss])

model.summary()

""" Setup model checkpoint """

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

# Alternatively, load the weights directly: model.load_weights(save_model_path)
try:
  model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})
except:
  print("model not loaded")
  pass

""" Training """
history = model.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                   epochs=epochs,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_val_examples / float(batch_size))),
                   callbacks=[cp])

model.save_weights(save_model_path)
with open(save_loss_path+"history", 'wb') as handle: # saving the history 
        dump(history.history, handle)

