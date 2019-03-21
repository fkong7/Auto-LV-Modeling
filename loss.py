import tensorflow as tf
from tensorflow.python.keras import losses


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
  
  
def dice_loss(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), 8)
    loss = 0.
    weights = [0.5,1.,1.5,1.,1.,1.,1.,1.]
    for i in range(8):
      loss += weights[i]*(1 - dice_coeff(y_true_one_hot[:,:,:,:,i], y_pred[:,:,:,i]))
    return loss
  
  
def bce_dice_loss(y_true, y_pred):
    loss = losses.sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
  
  
def la_loss(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), 8)
    loss = 1 - dice_coeff(y_true_one_hot[:,:,:,:,2], y_pred[:,:,:,2])
    return loss