#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:18:20 2019

@author: john
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

# Let's play around with transpose convolution on examples first
def test_conv2d_transpose(img_size, filter_size):
    print("Transpose convolution test for img_size={}, filter_size={}:".format(img_size, filter_size))
    
    x = (np.arange(img_size ** 2, dtype=np.float32) + 1).reshape((1, img_size, img_size, 1))
    f = (np.ones(filter_size ** 2, dtype=np.float32)).reshape((filter_size, filter_size, 1, 1))

    s = reset_tf_session()
    
    conv = tf.nn.conv2d_transpose(x, f, 
                                  output_shape=(1, img_size * 2, img_size * 2, 1), 
                                  strides=[1, 2, 2, 1], 
                                  padding='SAME')

    result = s.run(conv)
    print("input:")
    print(x[0, :, :, 0])
    print("filter:")
    print(f[:, :, 0, 0])
    print("output:")
    print(result[0, :, :, 0])
    s.close()
        
#test_conv2d_transpose(img_size=2, filter_size=2)
#test_conv2d_transpose(img_size=2, filter_size=3)
#test_conv2d_transpose(img_size=4, filter_size=2)
#test_conv2d_transpose(img_size=4, filter_size=3)

###############################################################################
# Let's play around with transpose convolution on examples first

vec_size=32
filter_size=2

print("Transpose convolution test for img_size={}, filter_size={}:".format(vec_size, filter_size))

img_size = vec_size

x2d = (np.arange(img_size ** 2, dtype=np.float32) + 1).reshape((1, img_size, img_size, 1))
f2d = (np.ones(filter_size ** 2, dtype=np.float32)).reshape((filter_size, filter_size, 1, 1))
    
x_ori = (np.arange(vec_size, dtype=np.float32) + 1).reshape((1, vec_size, 1))
f_ori = (np.ones(filter_size, dtype=np.float32)).reshape((filter_size, 1, 1))

f_ori = (np.ones(filter_size, dtype=np.float32)).reshape((filter_size, 1, 1))

s = reset_tf_session()


x = tf.convert_to_tensor(x_ori)
f = tf.convert_to_tensor(f_ori)

#conv = tf.contrib.nn.conv1d_transpose(x, f, (1, vec_size * 2, 1), 1)

stride = 4
#conv = tf.contrib.nn.conv1d_transpose(x, f, (1, vec_size*2, 1), stride)
conv = tf.contrib.nn.conv1d_transpose(x, f, (1, vec_size*2, 1), stride,padding='VALID')

result = s.run(conv)
print("input:")
print(x[0, :, :, 0])
print("filter:")
print(f[:, :, 0, 0])
print("output:")
print(result[0, :, :, 0])
s.close()
 
'''
tf.contrib.nn.conv1d_transpose(
    value,
    filter,
    output_shape,
    stride,
    padding='SAME',
    data_format='NWC',
    name=None
)


value: A 3-D Tensor of type float and shape [batch, in_width, in_channels] for NWC data format or [batch, in_channels, in_width] for NCW data format.

filter: A 3-D Tensor with the same type as value and shape [filter_width, output_channels, in_channels]. 
filter's in_channels dimension must match that of value.

output_shape: A 1-D Tensor representing the output shape of the deconvolution op.

stride: An integer. The number of entries by which the filter is moved right at each step.

padding: A string, either 'VALID' or 'SAME'. 
The padding algorithm. 
See the "returns" section of tf.nn.convolution for details.

data_format: A string. 'NHWC' and 'NCHW' are supported.

name: Optional name for the returned tensor.


'''       

#test_conv1d_transpose(vec_size=2, filter_size=2)
#test_conv2d_transpose(img_size=2, filter_size=3)
#test_conv2d_transpose(img_size=4, filter_size=2)
#test_conv2d_transpose(img_size=4, filter_size=3)