import os
import numpy as np
import glob
try:
    import tensorflow as tf
except Exception as e: print(e)
import SimpleITK as sitk
import vtk

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

def load_vtk_image(fn):
    """
    This function imports image file as vtk image.
    Args:
        fn: filename of the image data
    Return:
        label: label map as a vtk image
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    _, ext = fn.split(os.extsep, 1)

    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label = reader.GetOutput()
    elif ext=='nii' or ext=='nii.gz':
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(fn)
        reader.Update()
        image = reader.GetOutput()
        matrix = reader.GetQFormMatrix()
        if matrix is None:
            matrix = reader.GetSFormMatrix()
        matrix.Invert()
        Sign = vtk.vtkMatrix4x4()
        Sign.Identity()
        Sign.SetElement(0, 0, -1)
        Sign.SetElement(1, 1, -1)
        M = vtk.vtkMatrix4x4()
        M.Multiply4x4(matrix, Sign, M)
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToLinear()
        reslice.SetOutputSpacing(np.min(image.GetSpacing())*np.ones(3))
        reslice.Update()
        label = reslice.GetOutput()
        py_label = vtk_to_numpy(label.GetPointData().GetScalars())
        py_label = (py_label + reader.GetRescaleIntercept())/reader.GetRescaleSlope()
        label.GetPointData().SetScalars(numpy_to_vtk(py_label))
    else:
        raise IOError("File extension is not recognized: ", ext)
    return label

def writeVTKImage(vtkIm, fn):
    """
    This function writes a vtk image to disk
    Args:
        vtkIm: the vtk image to write
        fn: file name
    Returns:
        None
    """
    print("Writing vti with name: ", fn)

    _, extension = os.path.splitext(fn)
    if extension == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif extension == '.mhd':
        writer = vtk.vtkMetaImageWriter()
    else:
        raise ValueError("Incorrect extension " + extension)
    writer.SetInputData(vtkIm)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def vtk_write_mask_as_nifty(mask, image_fn, mask_fn):
    import vtk
    origin = mask.GetOrigin()
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_fn)
    reader.Update()
    writer = vtk.vtkNIFTIImageWriter()
    Sign = vtk.vtkMatrix4x4()
    Sign.Identity()
    Sign.SetElement(0, 0, -1)
    Sign.SetElement(1, 1, -1)
    M = reader.GetQFormMatrix()
    if M is None:
        M = reader.GetSFormMatrix()
    M2 = vtk.vtkMatrix4x4()
    M2.Multiply4x4(Sign, M, M2)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(mask)
    reslice.SetResliceAxes(M2)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    mask = reslice.GetOutput()
    mask.SetOrigin([0.,0.,0.])

    writer.SetInputData(mask)
    writer.SetFileName(mask_fn)
    writer.SetQFac(reader.GetQFac())
    q_mat = reader.GetQFormMatrix()
    writer.SetQFormMatrix(q_mat)
    s_mat = reader.GetSFormMatrix()
    writer.SetSFormMatrix(s_mat)
    writer.Write()
    return

def get_array_from_vtkImage(image):
    from vtk.util.numpy_support import vtk_to_numpy
    py_im = vtk_to_numpy(image.GetPointData().GetScalars())
    print(np.min(py_im),np.max(py_im))
    x , y, z = image.GetDimensions()
    out_im = py_im.reshape(z, y, x)
    return out_im

def get_vtkImage_from_array(py_im):
    from vtk.util.numpy_support import numpy_to_vtk
    z, y, x = py_im.shape
    vtkArray = numpy_to_vtk(py_im.flatten())
    image = vtk.vtkImageData()
    image.SetDimensions(x, y, z)
    image.GetPointData().SetScalars(vtkArray)
    return image
