# -*- coding: utf-8 -*-
import os
import glob
import functools
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import random

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import SimpleITK as sitk

from utils import getTrainNLabelNames
from preProcess import swapLabels
from preProcess import RescaleIntensity
from utils import np_to_tfrecords

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
from loss import bce_dice_loss, weighted_bce_dice_loss

from tensorflow.python.keras.optimizers import Adam

from pickle import dump
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image',  help='Name of the folder containing the image data')
parser.add_argument('--val',  help='Name of the folder containing the image data')
parser.add_argument('--attr', help='Attribute name of the folders containing 2d slices')
parser.add_argument('--output',  help='Name of the output folder')
parser.add_argument('--view', type=int, help='view id, axial(0), coronal(1), sagittal(2)')
parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
parser.add_argument('--num_epoch', type=int, help='Maximum number of epochs to run')
parser.add_argument('--num_class', type=int, help='Number of classes')
parser.add_argument('--channel', type=int, default=1, help='Number of channels of input images')
parser.add_argument('--size', type=int, default=[256, 256], nargs='+', help='Desired size of the input images')
parser.add_argument('--seed', type=int, default=41, help='Randome seed')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--lr', type=float, help='Learning rate')
args = parser.parse_args()

img_shape = (args.size[0], args.size[1], args.channel)

view_names = ['axial', 'coronal', 'sagittal']
data_folder_out = [ os.path.join(args.image, '2d_multiclass-%s%s_train' % (args.attr,view_names[args.view]))]*2
data_val_folder_out = [os.path.join(args.val, '2d_multiclass-%s%s_val' % (args.attr,view_names[args.view]))]*2
if args.channel>1:
   data_folder_out += '_multi%d' % args.channel
   data_val_folder_out += '_multi%d' % args.channel

save_model_path = os.path.join(args.output, 'weights_multi-all-%s_small2.hdf5' % view_names[args.view])
save_loss_path = os.path.join(args.output, view_names[args.view])

""" Create new directories """
try:
    os.makedirs(os.path.dirname(save_model_path))
    os.makedirs(os.path.dirname(save_loss_path))
except Exception as e: print(e)


""" Pre-process data """
def buildImageDataset(data_folder_out, modality, seed):
    x_train_filenames = []
    filenames = [None]*len(modality)
    nums = np.zeros(len(modality))
    for i, m in enumerate(modality):
      filenames[i], _ = getTrainNLabelNames(data_folder_out[i], m, ext='*.tfrecords')
      nums[i] = len(filenames[i])
      x_train_filenames+=filenames[i]
      #shuffle
      random.shuffle(x_train_filenames)
      
    print("Number of images obtained for training and validation: " + str(nums))
    
    """ Sample dataset """
    np.random.seed(seed)
    nums = np.max(nums) - nums
    for i , _ in enumerate(modality):
      index = sample(list(range(len(filenames[i]))), nums[i])
      x_train_filenames+=[filenames[i][j] for j in index]
    
    #x_train_filenames = bootstrapping(x_train_filenames)

    return x_train_filenames

x_train_filenames = buildImageDataset(data_folder_out, args.modality, args.seed)
x_val_filenames = buildImageDataset(data_val_folder_out, args.modality, args.seed)
print("Number of training examples after sampling: {}".format(len(x_train_filenames)))
print("Number of validation examples after sampling: {}".format(len(x_val_filenames)))

if len(x_val_filenames) ==0:
    x_train_filenames, x_val_filenames = train_test_split(x_train_filenames, test_size=0.2, random_state=args.seed)
    print("Number of training examples after sampling: {}".format(len(x_train_filenames)))
    print("Number of validation examples after sampling: {}".format(len(x_val_filenames)))
    
num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
#total_images = 0
#for f_i, file in enumerate(x_train_filenames): 
#    print(f_i, file) 
#    total_images += sum([1 for _ in tf.python_io.tf_record_iterator(file)])

"""## Set up train and validation datasets
Note that we apply image augmentation to our training dataset but not our validation dataset.
"""

tr_cfg = {
    #'num_class': args.num_class,
    'resize': [img_shape[0], img_shape[1]],
    'horizontal_flip': True,
    #'rotation': 10.,
    'changeIntensity': {"scale": [0.9, 1.1],"shift": [-0.1, 0.1]}, 
    'width_shift_range': 0.15,
    'height_shift_range': 0.15
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'resize': [img_shape[0], img_shape[1]]
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

train_ds = get_baseline_dataset(x_train_filenames, preproc_fn=tr_preprocessing_fn,
                                batch_size=args.batch_size)
val_ds = get_baseline_dataset(x_val_filenames, preproc_fn=val_preprocessing_fn,
                              batch_size=args.batch_size)

"""# Build the model"""
inputs, outputs = UNet2D(img_shape, args.num_class)
model = models.Model(inputs=[inputs], outputs=[outputs])

adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dice_loss])
model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dice_loss])

model.summary()

""" Setup model checkpoint """

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_loss', factor=0.8, patience=5, min_lr=0.000005)
erly_stp = tf.keras.callbacks.EarlyStopping(monitor='val_dice_loss', patience=30)
# Alternatively, load the weights directly: model.load_weights(save_model_path)
try:
  model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})
except:
  print("model not loaded")
  pass

""" Training """
history = model.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(args.batch_size))),
                   epochs=args.num_epochs,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_val_examples / float(args.batch_size))),
                   callbacks=[cp, lr_schedule, erly_stp])


model.save_weights(save_model_path)
with open(save_loss_path+"history", 'wb') as handle: # saving the history 
        dump(history.history, handle)

