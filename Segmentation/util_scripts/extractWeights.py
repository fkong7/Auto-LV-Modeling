import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from loss import dice_loss, bce_dice_loss
import os

def saveWeights(model_fn,save_fn):
    model = models.load_model(model_fn, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})
    model.save_weights(save_fn)
    del model
    K.clear_session()

if __name__ == '__main__':
    view_names = ['axial', 'coronal', 'sagittal']
    dir_path = '/global/scratch/fanwei_kong/2DUNet/Logs'
    #base_name = ['MMWHS_small_btstrp','MMWHS_small_btstrp2','MMWHS_small_btstrp3']
    base_name = ['MMWHS_2/total_run3']
    for folder in base_name:
        for view in view_names:
            model_name =  'multi-all-%s_small2.hdf5' % view
            model_path = os.path.join(dir_path, folder, model_name)
            save_name = 'weights_' + model_name
            save_path = os.path.join(dir_path, folder, save_name)
            saveWeights(model_path, save_path)


