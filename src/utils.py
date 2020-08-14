import os
import numpy as np
import glob
try:
    import tensorflow as tf
except Exception as e: print(e)
import SimpleITK as sitk


def sample_in_range(range_):
  return (range_[1] - range_[0]) * np.random.random_sample() + range_[0]

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz',fn='_train', post='_masks'):
  x_train_filenames = []
  y_train_filenames = []
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+fn,ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  try:
      for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+fn+post,ext))):
          y_train_filenames.append(os.path.realpath(subject_dir))
  except Exception as e: print(e)

  return x_train_filenames, y_train_filenames

def writeIm(fn, image):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fn)
    writer.Execute(image)
    return 

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def np_to_tfrecords(X, Y, file_path_prefix=None, Prob=None, verbose=True, debug=True):
            
    if Y is not None:
        assert X.shape[1:] == Y.shape
    if Prob is not None:
        assert X.shape == Prob.shape[:-1]

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())

    if debug:
        print("**** X ****")
        print(X.shape, X.flatten().shape)
        print(X.dtype)
    if Y is not None:
        d_feature['Y'] = _int64_feature(Y.flatten())
        if debug:
            print("**** Y shape ****")
            print(Y.shape, Y.flatten().shape)
            print(Y.dtype)
    if Prob is not None:
        d_feature['P'] = _float_feature(Prob.flatten())
        if debug:
            print("**** P shape ****")
            print(Prob.shape, Prob.flatten().shape)
            print(Prob.dtype)

    #first axis is the channel dimension
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])    
    d_feature['shape2'] = _int64_feature([X.shape[2]])

    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))
        
