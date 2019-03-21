import os
import numpy as np
import glob

def getTrainNLabelNames(data_folder, m, ext='*.nii.gz'):
  x_train_filenames = []
  y_train_filenames = []
  for subject_dir in sorted(glob.glob(os.path.join(data_folder,m+'_train',ext))):
      x_train_filenames.append(os.path.realpath(subject_dir))
  for subject_dir in sorted(glob.glob(os.path.join(data_folder ,m+'_train_masks',ext))):
      y_train_filenames.append(os.path.realpath(subject_dir))
  return x_train_filenames, y_train_filenames


def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            
    if Y is not None:
        assert X.shape == Y.shape
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing example into {}".format(result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    
    d_feature = {}
    d_feature['X'] = _float_feature(X.flatten())
    if Y is not None:
        d_feature['Y'] = _int64_feature(Y.flatten())
    d_feature['shape0'] = _int64_feature([X.shape[0]])
    d_feature['shape1'] = _int64_feature([X.shape[1]])
            
    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))
        