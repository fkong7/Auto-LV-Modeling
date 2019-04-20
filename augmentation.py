import tensorflow as tf
import tensorflow.contrib as tfcontrib

def shift_img(output_img, label_img, width_shift_range, height_shift_range):
  """This fn will perform the horizontal or vertical shift"""
  img_shape = output_img.get_shape().as_list()
  if width_shift_range or height_shift_range:
      if width_shift_range:
        width_shift_range = tf.random_uniform([], 
                                              -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
      if height_shift_range:
        height_shift_range = tf.random_uniform([],
                                               -height_shift_range * img_shape[0],
                                               height_shift_range * img_shape[0])
      # Translate both 
      output_img = tfcontrib.image.translate(output_img,
                                             [width_shift_range, height_shift_range])
      label_img = tfcontrib.image.translate(label_img,
                                             [width_shift_range, height_shift_range])
  return output_img, label_img

"""## Flipping the image randomly"""

def flip_img(horizontal_flip, tr_img, label_img):
  if horizontal_flip:
    flip_prob = tf.random_uniform([], 0.0, 1.0)
    tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
  return tr_img, label_img

"""##Scale/shift the image intensity randomly"""

def changeIntensity_img(tr_img, label_img, changeIntensity=False):
  if changeIntensity:
    scale = tf.random_uniform([], 0.9, 1.1)
    shift = tf.random_uniform([], -0.1, 0.1)
    tr_img = tr_img*scale+shift
    
  return tr_img, label_img

"""## Assembling our transformations into our augment function"""

def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             horizontal_flip=False,  # Random left right flip,
             changeIntensity=False,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize_images(label_img, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    img = tf.image.resize_images(img, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
 
  img, label_img = flip_img(horizontal_flip, img, label_img)
  img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
  img, label_img = changeIntensity_img(img, label_img,changeIntensity )
  return img, label_img

