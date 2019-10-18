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
"""## Rotate the image with random angle""" 
def rotate_img(rotation, tr_img, label_img):
    if rotation:
        angle = rotation * 3.1415926 / 180.
        angle_rand = tf.random_uniform([], -1*angle, angle)
        tf_img = tf.contrib.image.rotate(tr_img, angle_rand)
        label_img = tf.contrib.image.rotate(label_img, angle_rand)
    return tr_img, label_img

"""##Scale/shift the image intensity randomly"""

def changeIntensity_img(tr_img, label_img, change):
  if change:
    scale = tf.random_uniform([], change['scale'][0], change['scale'][1])
    shift = tf.random_uniform([], change['shift'][0], change['shift'][1])
    tr_img = tr_img*scale+shift
    tr_img = tf.clip_by_value(tr_img, -1., 1.)
  return tr_img, label_img

"""##Scale the image intensity for different cardiac structures"""
def change_class_intensity_img(tr_img, label_img, change, num_class):
    if change and num_class:
        for i in range(1, num_class):
            scale = tf.random_uniform([], change['scale'][0], change['scale'][1])
            shift = tf.random_uniform([], change['shift'][0], change['shift'][1])
            scaled = tr_img*scale+shift
            tr_img = tf.where(tf.equal(label_img,i), scaled, tr_img)
    return tr_img, label_img
"""## Assembling our transformations into our augment function"""

def _augment(img,
             label_img,
             num_class =0, 
             resize=None,  # Resize the image to some size e.g. [256, 256]
             horizontal_flip=False,  # Random left right flip,
             rotation = 0, 
             changeIntensity=False,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize_images(label_img, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    img = tf.image.resize_images(img, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
  img, label_img = flip_img(horizontal_flip, img, label_img)
  img, label_img = rotate_img(rotation, img, label_img)
  img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
  img, label_img = changeIntensity_img(img, label_img,changeIntensity )
  img, label_img = change_class_intensity_img(img, label_img, changeIntensity, num_class) 
  return img, label_img

