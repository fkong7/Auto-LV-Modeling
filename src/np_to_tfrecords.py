import numpy as np
import tensorflow as tf

__author__ = "Sangwoong Yoon"

def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """
    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
            

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
    d_feature['shape0'] = _int64_feature(X.shape[0])
    d_feature['shape1'] = _int64_feature(X.shape[1])
       
            
    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

        
#################################    
##      Test and Use Cases     ##
#################################

# 1-1. Saving a dataset with input and label (supervised learning)
#xx = np.random.randn(10,5)
#yy = np.random.randn(10,1)
#np_to_tfrecords(xx, yy, 'test1', verbose=True)

# 1-2. Check if the data is stored correctly
# open the saved file and check the first entries
#for serialized_example in tf.python_io.tf_record_iterator('test1.tfrecords'):
#    example = tf.train.Example()
#    example.ParseFromString(serialized_example)
#    x_1 = np.array(example.features.feature['X'].float_list.value)
#    y_1 = np.array(example.features.feature['Y'].float_list.value)
#    break
    
# the numbers may be slightly different because of the floating point error.
#print xx[0]
#print x_1
#print yy[0]
#print y_1


# 2. Saving a dataset with only inputs (unsupervised learning)
#xx = np.random.randn(100,100)
#np_to_tfrecords(xx, None, 'test2', verbose=True)