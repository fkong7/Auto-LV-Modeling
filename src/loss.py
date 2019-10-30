import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K



def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
  
  
def dice_loss(y_true, y_pred):
    num_class = y_pred.get_shape().as_list()[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
    loss = 0.
    for i in range(num_class):
      loss += (1 - dice_coeff_mean(y_true_one_hot[:,:,:,:,i], y_pred[:,:,:,i]))
    return loss
  
def dice_coeff_mean(y_true, y_pred):
    smooth = 1.
    # Flatten
    shape = tf.shape(y_pred)
    batch = shape[0]
    length = tf.reduce_prod(shape[1:])
    y_true_f = tf.reshape(y_true, [batch,length])
    y_pred_f = tf.reshape(y_pred, [batch,length])
    intersection = tf.reduce_sum(tf.multiply(y_true_f ,y_pred_f), axis=-1)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1) + smooth)
    return tf.reduce_mean(score)

def bce_dice_loss(y_true, y_pred):
    loss = losses.sparse_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
  
  
def la_loss(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), 8)
    loss = 1 - dice_coeff(y_true_one_hot[:,:,:,:,2], y_pred[:,:,:,2])
    return loss

def weighted_categorical_crossentropy(weights):
    #weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        num_class = y_pred.get_shape().as_list()[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
        #loss = 0.
        #for i in range(num_class):
        #    loss += tf.reduce_mean(losses.categorical_crossentropy(tf.squeeze(y_true_one_hot[:,:,:,:,i]), y_pred[:,:,:,i])) * weights[i]
        loss = tf.squeeze(y_true_one_hot) * K.log(y_pred)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]), axis=0)*weights
        loss = - K.sum(loss, -1)
        return loss
    return loss

def weighted_dice(weights):
    def loss(y_true, y_pred):
        num_class = y_pred.get_shape().as_list()[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
        loss = 0.
        for i in range(num_class):
            loss += weights[i] * (1 - dice_coeff_mean(y_true_one_hot[:,:,:,:,i], y_pred[:,:,:,i]))
        return loss
    return loss

def weighted_bce_dice_loss(weights):
    #weights = K.variable(weights)
    def loss(y_true, y_pred):
        #loss = weighted_categorical_crossentropy(weights)(y_true, y_pred) + weighted_dice(weights)(y_true, y_pred)
        loss = weighted_categorical_crossentropy(weights)(y_true, y_pred)
        return loss
    return loss


