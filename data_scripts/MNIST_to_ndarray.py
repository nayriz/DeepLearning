# hacked from 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)


import gzip
import numpy
from tensorflow.python.platform import gfile

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):

  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data



def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):

  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

one_hot = False
with gfile.Open('train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)

with gfile.Open('train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

with gfile.Open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)

with gfile.Open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

numpy.save('X_train_MNIST',train_images)
numpy.save('y_train_MNIST',train_labels)
numpy.save('X_test_MNIST',test_images)    
numpy.save('y_test_MNIST',test_labels)    
