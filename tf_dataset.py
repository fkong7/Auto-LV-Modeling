import tensorflow as tf
import functools
from augmentation import _augment

def _parse_function(example_proto):
  features = {"X": tf.VarLenFeature(tf.float32),
              "Y": tf.VarLenFeature(tf.int64),
              "shape0": tf.FixedLenFeature((), tf.int64),
              "shape1": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.sparse_tensor_to_dense(parsed_features["X"])
  height = tf.cast(parsed_features["shape0"], tf.int32)
  width = tf.cast(parsed_features["shape1"], tf.int32)
  print(tf.shape(img))
  with tf.Session() as sess:
      h, w = sess.run([height,width])
  print(h)
  print(w)
  label = tf.sparse_tensor_to_dense(parsed_features["Y"])
  img = tf.reshape(img, tf.stack([height, width,1]))
  label = tf.reshape(label, tf.stack([height, width,1]) )
  label = tf.cast(label, tf.int32)
  return img, label

def _process_pathnames(fname):
  # We map this function onto each pathname pair  
  dataset_str = tf.read_file(fname)
  dataset = tf.data.TFRecordDataset(dataset_str)
  parsed_features = dataset.map(_parse_function)
  
  iterator = dataset.make_one_shot_iterator()
  #data = iterator.get_next()
  
  img = tf.sparse_tensor_to_dense(parsed_features["X"])
  height = tf.cast(parsed_features["shape0"], tf.int32)
  width = tf.cast(parsed_features["shape1"], tf.int32)
  label = tf.sparse_tensor_to_dense(parsed_features["Y"])
  img = tf.reshape(img, tf.stack([height, width]))
  label = tf.reshape(label, tf.stack([height, width]) )

  return img, label

def get_baseline_dataset(filenames, preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=10,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  files = tf.data.Dataset.from_tensor_slices(filenames)

  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=threads))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_parse_function, num_parallel_calls=threads)
  # dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert batch_size == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  print(num_x)
  if shuffle:
    dataset = dataset.shuffle(6000)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  dataset = dataset.prefetch(buffer_size=batch_size)

  return dataset

