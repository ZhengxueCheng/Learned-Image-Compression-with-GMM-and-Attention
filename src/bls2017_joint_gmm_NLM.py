# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.
This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704
Modified by Victor Xing (11/15/2018)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Dependency imports
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.summary import summary
import tensorflow.distributions as tfd
import math
from tensorflow_compression.python.ops import coder_ops
from tensorflow_compression.python.ops import math_ops as tfc_math_ops
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

import glob
import functools
import time
import os
import shutil
import sys
from tensorflow.python.client import timeline

# To remove the "iCCP incorrect RGB profile" warning display
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

# Parameters used in the training phase, change to your preference
#EVAL_FREQUENCY = 500
MDATA_FREQUENCY = 500
SAVE_FREQUENCY = 1000
VISUALIZE_FREQUENCY = 1000

# Your number of processors, for parallel preprocessing
NUM_THREADS = 8#48


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='g')
            if sub_sample:
                g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='phi')
            if sub_sample:
                phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='theta')


        g_x = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
        theta_x = tf.reshape(theta, [tf.shape(theta)[0], -1, tf.shape(theta)[-1]])
        phi_x = tf.reshape(phi, [tf.shape(phi)[0], -1, tf.shape(phi)[-1]])
        phi_x = tf.transpose(phi_x, [0,2,1])

        f = tf.matmul(theta_x, phi_x)       
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        
        #y = tf.reshape(y, [batchsize, height, width, out_channels])
        y = tf.reshape(y, tf.shape(g))
        
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = slim.batch_norm(w_y)
        z = input_x + w_y
        return z

def residualblock(tensor, num_filters, scope="residual_block"):
  """Builds the residual block"""
  with tf.variable_scope(scope):
    with tf.variable_scope("conv0"):
      layer = tfc.SignalConv2D(
        num_filters//2, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu)
      output = layer(tensor)

    with tf.variable_scope("conv1"):
      layer = tfc.SignalConv2D(
        num_filters//2, (3,3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu)
      output = layer(output)

    with tf.variable_scope("conv2"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None)
      output = layer(output)
      
    tensor = tensor + output
       
  return tensor


def NonLocalAttentionBlock(input_x, num_filters, scope="NonLocalAttentionBlock"):
  """Builds the non-local attention block"""
  with tf.variable_scope(scope):
    trunk_branch = residualblock(input_x, num_filters, scope="trunk_RB_0")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_1")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_2")
    
    #attention_branch = NonLocalBlock(input_x, num_filters//2, sub_sample=False, is_bn=False)
    #attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_0")
    # ====Revision: REMOVE the Non Local Block============
    attention_branch = residualblock(input_x, num_filters, scope="attention_RB_0")
    '''
    with tf.variable_scope("downscale"):
      layer = tfc.SignalConv2D(
        num_filters, (3,3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=None)
      attention_branch = layer(attention_branch)
    '''
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_1")
    '''
    with tf.variable_scope("upscale"):
      layer = tfc.SignalConv2D(
          num_filters, (3,3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=None)
      attention_branch = layer(attention_branch)
    ''' 
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_2")

    with tf.variable_scope("conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None)
      attention_branch = layer(attention_branch)
    attention_branch = tf.sigmoid(attention_branch)
  
  tensor = input_x + tf.multiply(attention_branch, trunk_branch)
  return tensor



def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  kernel_size = 3
  #Use three 3x3 filters to replace one 9x9
  
  with tf.variable_scope("analysis"):

    # Four down-sampling blocks
    for i in range(4):
      if i > 0:
        with tf.variable_scope("Block_" + str(i) + "_layer_0"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tf.nn.leaky_relu)
          tensor2 = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_1"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu)
          tensor2 = layer(tensor2)
        
        tensor = tensor + tensor2


      if i < 3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):
          shortcut = tfc.SignalConv2D(num_filters, (1, 1), corr=True, strides_down=2, padding="same_zeros",
                                      use_bias=True, activation=None)
          shortcut_tensor = shortcut(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu)
          tensor = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN())
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

        if i == 1:
          #Add one NLAM
          before = tensor
          tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")
          after = tensor
          

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
            use_bias=False, activation=None) 
          tensor = layer(tensor)

        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")
        

    return tensor

def hyper_analysis(tensor, num_filters):
  """Build the analysis transform in hyper"""

  with tf.variable_scope("hyper_analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters     
    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters 
    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  kernel_size = 3
  #Use four 3x3 filters to replace one 9x9
  
  with tf.variable_scope("synthesis"):

    # Four up-sampling blocks
    for i in range(4):
      if i == 0:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")

      if i == 2:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")
        
      with tf.variable_scope("Block_" + str(i) + "_layer_0"):
        layer = tfc.SignalConv2D(
          num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
        tensor2 = layer(tensor)

      with tf.variable_scope("Block_" + str(i) + "_layer_1"):
        layer = tfc.SignalConv2D(
          num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
        tensor2 = layer(tensor2)
        tensor = tensor + tensor2


      if i <3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):

          # Use Sub-Pixel to replace deconv.
          shortcut = tfc.SignalConv2D(num_filters*4, (1, 1), corr=False, strides_up=1, padding="same_zeros",
                                      use_bias=True, activation=None)
          shortcut_tensor = shortcut(tensor)
          shortcut_tensor = tf.depth_to_space(shortcut_tensor, 2)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):

          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(num_filters*4, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=tf.nn.leaky_relu)
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)         
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          
          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(12, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=None)
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)
          

    return tensor

def hyper_synthesis(tensor, num_filters):
  """Builds the hyper synthesis transform"""

  with tf.variable_scope("hyper_synthesis", reuse=tf.AUTO_REUSE):
    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters*2, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor

def masked_conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    scope="masked"):
  
  with tf.variable_scope(scope):
    mask_type = mask_type.lower()
    batch_size, height, width, channel = inputs.get_shape().as_list()

    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      "kernel height and width should be odd number"

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    if mask_type is not None:
      mask = np.ones(
        (kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

      mask[center_h, center_w+1: ,: ,:] = 0.
      mask[center_h+1:, :, :, :] = 0.

      if mask_type == 'a':
        mask[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    return outputs


  
def entropy_parameter(tensor, inputs, num_filters, training):
  """tensor: the output of hyper autoencoder (phi) to generate the mean and variance
     inputs: the variable needs to be encoded. (y)
  """
  with tf.variable_scope("entropy_parameter", reuse=tf.AUTO_REUSE):

    half = tf.constant(.5)

    if training:
      noise = random_ops.random_uniform(array_ops.shape(inputs), -half, half)
      values = math_ops.add_n([inputs, noise])

    else: #inference
      if inputs is not None: #compress
        values = math_ops.round(inputs)
        
    #Masked CNN 5x5
    #input: values, tensor = tf.concat(masked, tensor)
    masked = masked_conv2d(values, num_filters*2, [5, 5], "A", scope='masked')
    tensor = tf.concat([masked, tensor], axis=3)
      

    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          640, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          640, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters*9, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    #=======Single distribution model========
    '''
    mean, variance = tf.split(tensor, num_or_size_splits=2, axis = 3)

    variance = math_ops.abs(variance)

    dist = tfd.Normal(loc = mean, scale = variance)
    '''
    #=========Gaussian Mixture Model=========
    prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
             tf.split(tensor, num_or_size_splits=9, axis = 3)
    scale0 = math_ops.abs(scale0)
    scale1 = math_ops.abs(scale1)
    scale2 = math_ops.abs(scale2)

    dist_0 = tfd.Normal(loc = mean0, scale = scale0, name='dist_0')
    dist_1 = tfd.Normal(loc = mean1, scale = scale1, name='dist_1')
    dist_2 = tfd.Normal(loc = mean2, scale = scale2, name='dist_2')

    probs = tf.stack([prob0, prob1, prob2], axis=-1)
    probs = tf.nn.softmax(probs, axis=-1)
  
    # To merge them together
    means = tf.stack([mean0, mean1, mean2], axis=-1)
    variances = tf.stack([scale0, scale1, scale2], axis=-1)

    # =======================================
    
    #Calculate the likelihoods for inputs
    if inputs is not None:

      #=======Single distribution model========
      '''
      lower = dist.cdf(values - half)
      upper = dist.cdf(values + half)
      likelihood_bound = tf.constant(1e-9)
      likelihoods = tf.maximum(upper-lower, likelihood_bound)
      '''
      #=========Gaussian Mixture Model=========
      likelihoods_0 = dist_0.cdf(values + half) - dist_0.cdf(values - half)
      likelihoods_1 = dist_1.cdf(values + half) - dist_1.cdf(values - half)
      likelihoods_2 = dist_2.cdf(values + half) - dist_2.cdf(values - half)

      likelihoods = probs[:,:,:,:,0]*likelihoods_0 + probs[:,:,:,:,1]*likelihoods_1 + probs[:,:,:,:,2]*likelihoods_2

      # =======REVISION: Robust version ==========
      edge_min = probs[:,:,:,:,0]*dist_0.cdf(values + half) + \
                 probs[:,:,:,:,1]*dist_1.cdf(values + half) + \
                 probs[:,:,:,:,2]*dist_2.cdf(values + half)
      
      edge_max = probs[:,:,:,:,0]* (1.0 - dist_0.cdf(values - half)) + \
                 probs[:,:,:,:,1]* (1.0 - dist_1.cdf(values - half)) + \
                 probs[:,:,:,:,2]* (1.0 - dist_2.cdf(values - half))
      likelihoods = tf.where(values < -254.5, edge_min, tf.where(values > 255.5, edge_max, likelihoods))

      
      likelihood_lower_bound = tf.constant(1e-6)
      likelihood_upper_bound = tf.constant(1.0)
      likelihoods = tf.minimum(tf.maximum(likelihoods, likelihood_lower_bound), likelihood_upper_bound)
      
    else:
      values = None
      likelihoods = None
        
  return values, likelihoods, means, variances, probs

  

def train_preprocess(image):
  """Performs data augmentation on the training set."""

  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
  image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
  image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
  image = tf.image.random_hue(image, max_delta=0.2)

  # Random cropping
  crop_shape = (args.patchsize, args.patchsize, 3)
  image = tf.random_crop(image, crop_shape)

  # Make sure the image is still in [0, 1]
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def train():
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files = glob.glob(args.train_glob)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.apply(
                      tf.contrib.data.shuffle_and_repeat(len(train_files)))
    train_dataset = train_dataset.map(load_image,
                                      num_parallel_calls=NUM_THREADS)
    train_dataset = train_dataset.map(train_preprocess,
                                      num_parallel_calls=NUM_THREADS)
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(1)

    # Need to crop validation and test images too to have the same size
    # Can probably use deterministic instead of random cropping...
    mapcrop = functools.partial(tf.random_crop,
                                size=(args.patchsize, args.patchsize, 3))

    valid_files = glob.glob(args.valid_glob)
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_dataset = valid_dataset.map(load_image,
                                      num_parallel_calls=NUM_THREADS)
    valid_dataset = valid_dataset.map(mapcrop,
                                      num_parallel_calls=NUM_THREADS)
    valid_dataset = valid_dataset.batch(args.batchsize)

    test_files = glob.glob(args.test_glob)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
    test_dataset = test_dataset.map(load_image,
                                    num_parallel_calls=NUM_THREADS)
    test_dataset = test_dataset.map(mapcrop,
                                    num_parallel_calls=NUM_THREADS)
    test_dataset = test_dataset.batch(args.batchsize)

    num_pixels = args.patchsize**2 * args.batchsize

  # Build reinitializable initializers.
  iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                         train_dataset.output_shapes)
  x = iter.get_next()
  train_init_op = iter.make_initializer(train_dataset)
  valid_init_op = iter.make_initializer(valid_dataset)
  test_init_op = iter.make_initializer(test_dataset)

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)

  # =====REVISION: Limit the range for robust version ==============
  y = tf.clip_by_value(y, -255, 256)

  # --------------Build a hyper autoencoder------------
  
  z = tf.stop_gradient(y)
  z_hat = hyper_analysis(z, args.num_filters)
  
  entropy_bottleneck = tfc.EntropyBottleneck()
  z_tilde, z_likelihoods = entropy_bottleneck(z_hat, training=True)
  
  phi = hyper_synthesis(z_tilde, args.num_filters)
  
  y_tilde, likelihoods, _, _, _ = entropy_parameter(phi, y, args.num_filters, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)

  bpp_z = tf.reduce_sum(tf.log(z_likelihoods)) / (-np.log(2) * num_pixels)

  # ----------------------------

  # Total number of bits divided by number of pixels.
  bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
  
  # Mean squared error across pixels.
  mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  mse *= 255 ** 2
  

  # MS-SSIM across pixels
  #_MSSSIM_WEIGHTS = [1., 1., 1., 1., 1.]
  _MSSSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
  msssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(x_tilde*255, x*255, 255, _MSSSIM_WEIGHTS))

  # The rate-distortion cost.
  #loss = args.lmbda * mse + bpp

  # ---------New rate-distortion cost.------------
  #loss = args.lmbda * mse + bpp + bpp_z

  # ---------New rate-distortion cost.------------
  loss = args.lmbda * msssim + bpp + bpp_z

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  main_step = main_optimizer.minimize(loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate*10.0)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  saver = tf.train.Saver()

  # Limit the GPU memory usage
  #config = tf.ConfigProto()  
  #config.gpu_options.per_process_gpu_memory_fraction = 0.4 
  #with tf.Session(config=config) as sess:
  with tf.Session() as sess:
    # Initialize summaries for Tensorboard
    # Plots: training and validation loss, training and validation rate (bpp)
    loss_summ = tf.summary.scalar("loss", loss)
    bpp_summ = tf.summary.scalar("bpp", bpp)
    val_loss, val_loss_update = tf.metrics.mean(loss, name="val_loss")
    val_bpp, val_bpp_update = tf.metrics.mean(bpp, name="val_bpp")
    val_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                 scope="val_loss") \
      + tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                          scope="val_bpp")
    val_loss_summ = tf.summary.scalar("loss", val_loss)
    val_bpp_summ = tf.summary.scalar("bpp", val_bpp)
    merged = tf.summary.merge([loss_summ, bpp_summ, val_loss_summ,
                               val_bpp_summ])

    # Initialize graph parameters
    sess.run(tf.variables_initializer(var_list=val_vars))
    sess.run(tf.global_variables_initializer())
    print('\nVariables initialized\n')
    print('Number of trainable parameters : {:d}\n'
          .format(np.sum([np.prod(v.get_shape().as_list())
                         for v in tf.trainable_variables()])))
    #print(tf.trainable_variables())

    n_train = len(glob.glob(args.train_glob))
    n_valid = len(glob.glob(args.valid_glob))

    print(n_train)
    print(n_valid)
    
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_writer = tf.summary.FileWriter(val_log_dir)
    step = 0
    start_time = epoch_time = time.time()
    
    # Load the latest model checkpoint to continue
    if os.path.isfile(args.checkpoint_dir + '/checkpoint'):
      latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
      import re
      step = int(next(re.finditer("(\d+)(?!.*\d)", latest)).group(0))
      saver.restore(sess, save_path=latest)
      print("[*] Load SUCCESS with " + latest)

    # Begin training loop
    for epoch in range(args.epochs):
      epoch_time = time.time()
      sess.run(train_init_op)

      fin = open("logs/Imagenet256_joint_lr" + str(args.learning_rate) + "_lmbd_" +
                 str(args.lmbda) + "_filter_" + str(args.num_filters)+"_4_3x3_v1_subpixel_Context5x5_MSSSIM_GMM_NLM.txt", 'a')    
      for _ in range(n_train // args.batchsize):
        step += 1

        sess.run(train_op)

        if step % VISUALIZE_FREQUENCY == VISUALIZE_FREQUENCY - 1:
            # Visualize the current estimated pdf of \tilde{y} in Tensorboard
            # (Images tab)
            vis_img = sess.run(entropy_bottleneck.visualize())
            train_writer.add_summary(vis_img, step)


        if step % SAVE_FREQUENCY == 0:
            saver.save(sess, args.checkpoint_dir+'/model', global_step=step)

        #print(step)
        PRINT_FREQUENCY = n_train // args.batchsize
        PIXEL_MAX = 255.
        if step % PRINT_FREQUENCY == 0:
          mse_value, loss_value, bpp_value, bpp_z_value = sess.run([mse, loss, bpp, bpp_z])
          psnr_value = 20 * math.log10( PIXEL_MAX / math.sqrt(mse_value))
          print('Iteration %d (epoch %d), Training loss: %.3f, Rate: %.3f, HyperRate: %.3f, MSE: %.3f, PSNR: %.2f dB' %
                (step, epoch, loss_value, bpp_value, bpp_z_value,  mse_value, psnr_value))
          fin.write('Iteration %d (epoch %d), Training loss: %.3f, Rate: %.3f, HyperRate: %.3f, MSE: %.3f, PSNR: %.2f dB' %
                    (step, epoch, loss_value, bpp_value, bpp_z_value,  mse_value, psnr_value))
          fin.write("\n")
        

      # Compute validation loss and bpp
      sess.run(valid_init_op)
      sess.run(tf.variables_initializer(var_list=val_vars))
      for _ in range(n_valid // args.batchsize):
         sess.run([val_loss_update, val_bpp_update])
      summary = sess.run(val_loss_summ)
      val_writer.add_summary(summary, step)
      summary = sess.run(val_bpp_summ)
      val_writer.add_summary(summary, step)
      elapsed_time = time.time() - epoch_time

      # Print info on training progression
      print('Time: %.1f s, Iteration %d (epoch %d), Validation loss: %.3f' % (elapsed_time, step, epoch, sess.run(val_loss)))      
      fin.write('Time: %.1f s, Iteration %d (epoch %d), Validation loss: %.3f' % (elapsed_time, step, epoch, sess.run(val_loss)))
      fin.write("\n")
      #print('Iteration %d (epoch %d)' % (step, epoch))
      #print('Training loss: %.3f' % sess.run(loss))
      #print('Validation loss: %.1f' % sess.run(val_loss))
      #print('%.1f s\n' % elapsed_time)
      #sys.stdout.flush()

    training_time = time.time() - start_time
    train_writer.close()
    val_writer.close()

    # Print the test loss and total training time
    sess.run(test_init_op)
    sess.run(tf.variables_initializer(var_list=val_vars))
    n_test = len(glob.glob(args.test_glob))
    for _ in range(n_test // args.batchsize):
        sess.run([val_loss_update, val_bpp_update])
    print('Test loss: %.1f' % sess.run(val_loss))
    print('Training time : %.1fs' % (training_time))


def compress():
  
  range_coder_precision = 16
  start = time.time()
  
  
  with tf.device('/cpu:0'):
    # Load input image and add batch dimension.
    x = load_image(args.input)

    # Pad the x to x_pad
    mod = constant_op.constant([64, 64, 1], dtype=tf.int32)
    div = tf.ceil(tf.truediv(tf.shape(x), mod)) 
    div = tf.cast(div, tf.int32)
    paddings = tf.subtract(tf.multiply(div, mod), tf.shape(x))
    paddings = tf.expand_dims(paddings, 1)
    paddings = tf.concat([tf.convert_to_tensor(np.zeros((3,1)), dtype=tf.int32), paddings], axis=1)

    x_pad = tf.pad(x, paddings, "REFLECT")
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    
    x_pad = tf.expand_dims(x_pad, 0)
    x_pad.set_shape([1, None, None, 3])

    # Transform and compress the image, then remove batch dimension.
    y = analysis_transform(x_pad, args.num_filters)

    # --------------Build a hyper autoencoder------------
    z = hyper_analysis(y, args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    string = entropy_bottleneck.compress(z)
    string = tf.squeeze(string, axis=0)

    z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

    string_rec = tf.expand_dims(string, 0)
    z_tilde = entropy_bottleneck.decompress(string_rec, tf.shape(z)[1:], channels=args.num_filters)

    phi = hyper_synthesis(z_tilde, args.num_filters)

    phi_again = hyper_synthesis(z_tilde, args.num_filters)

    #=====REVISION： for Gaussian Mixture Model======
    _, y_likelihoods, mean, variance, prob = entropy_parameter(phi, y, args.num_filters, training=False)
    y = tf.clip_by_value(y, -255, 256)
    y_hat = math_ops.round(y)

    '''
    # Transform the quantized image back (if requested).
    y_hat, likelihoods, mean, variance = entropy_parameter(phi, y, args.num_filters, training=False)


    # Largest distance observed between lower tail quantile and median,
    # or between median and upper tail quantile.
    minima = math_ops.reduce_max(-y_hat)
    maxima = math_ops.reduce_max(y_hat)
    minmax = math_ops.maximum(minima, maxima)
    minmax = math_ops.ceil(minmax)
    minmax = math_ops.maximum(minmax, 1)
    
    # Sample the density up to `minmax` around the median.
    samples = math_ops.range(-minmax, minmax + 1, dtype=tf.float32)
    half = constant_op.constant(.5, dtype=tf.float32)

    mean = tf.reshape(tf.layers.flatten(mean), [-1, 1])
    variance = tf.reshape(tf.layers.flatten(variance), [-1, 1])

    dist = tfd.Normal(loc=mean, scale=variance)

    samples = tf.reshape(samples, [1, -1])

    pmf = dist.cdf(samples + half) - dist.cdf(samples - half)

    # Add tail masses to first and last bin of pmf, as we clip values for
    # compression, meaning that out-of-range values get mapped to these bins.

    pmf = array_ops.concat([
      math_ops.add_n([pmf[:, :1], dist.cdf(-minmax - half)]),
      pmf[:, 1:-1],
      math_ops.add_n([pmf[:, -1:], 1 - dist.cdf(minmax + half)]),
    ], axis=-1)

    cdf = coder_ops.pmf_to_quantized_cdf(pmf, precision=range_coder_precision)
    cdf = array_ops.expand_dims(cdf, axis=1)

    values = tf.reshape(tf.layers.flatten(y_hat), [-1,1])
    values = math_ops.cast(values + minmax, dtypes.int16)

    string_y = tfc.range_encode(values, cdf, precision=range_coder_precision)
    '''
    


    x_hat = synthesis_transform(y_hat, args.num_filters)


    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

    # Total number of bits divided by number of pixels.
    eval_bpp = tf.reduce_sum(tf.log(y_likelihoods)) / (-np.log(2) * num_pixels) + \
               tf.reduce_sum(tf.log(z_likelihoods)) / (-np.log(2) * num_pixels)

    x_hat = x_hat[0, :tf.shape(x)[1], :tf.shape(x)[2], :]

    op = save_image('temp/temp.png', x_hat)
    
    # Mean squared error across pixels.
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    mse = tf.reduce_mean(tf.squared_difference(x * 255, x_hat))


    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # Load the latest model checkpoint, get the compressed string and the tensor
      # shapes.
      latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
      tf.train.Saver().restore(sess, save_path=latest)
      print(latest)

      tf.set_random_seed(1)
      '''
      
      string, x_shape, y_shape, string_y, minmax, y_hat = \
              sess.run([string, tf.shape(x), tf.shape(y), string_y, minmax, y_hat])

      #string, x_shape, y_shape, minmax, y_hat, mean, variance = \
      #        sess.run([string, tf.shape(x), tf.shape(y), minmax, y_hat, mean, variance])

      print(x_shape)
      print(y_shape)


      # Write a binary file with the shape information and the compressed string.
      with open(args.output, "wb") as f:
        f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
        f.write(np.array([len(string), minmax], dtype=np.uint16).tobytes())
        f.write(string)
        f.write(string_y)
      f.close()
      '''

      # If requested, transform the quantized image back and measure performance.
      if args.verbose:
        eval_bpp, mse, num_pixels, string = sess.run([eval_bpp, mse, num_pixels, string])
        sess.run(op)

        # The actual bits per pixel including overhead.
        #bpp = (8 + len(string) + len(string_y)) * 8 / num_pixels
        #bpp = os.path.getsize(args.output)* 8 / num_pixels

        bpp_side = len(string)*8/num_pixels

        psnr = sess.run(tf.image.psnr(x_hat, x*255, 255))
        msssim = sess.run(tf.image.ssim_multiscale(x_hat, x*255, 255))

        end = time.time()
        print("Time : {:0.3f}".format(end-start))
        
        print("Mean squared error: {:0.4}".format(mse))
        print("Information content of this image in bpp: {:0.4}".format(eval_bpp))
        #print("Actual bits per pixel for this image: {:0.4}".format(bpp))
        print("Side bits per pixel for z: {:0.4}".format(bpp_side))
        print("PSNR (dB) : {:0.4}".format(psnr[0]))
        print("MS-SSIM (dB) : {:0.4}".format(-10*np.log10(1-msssim[0])))

        fin = open("logs/bls2017.txt", 'a')
        fin.write("%.8f\t %.8f\t %.8f\t %.8f \t %.8f \t %.3f" % (mse, psnr[0], msssim[0], eval_bpp, bpp_side, (end-start) ))
        fin.write("\n")

        '''
        #=======REVISION: Visualize the Attention Module =======
        before, after = sess.run([before, after])
        print(before.shape)
        print(after.shape)

        vis_diff = np.var(before[0,:,:,:] - after[0,:,:,:], axis=(0,1))
        index = np.argmax(vis_diff)
        print('The channel with the largest difference before/after attention module is ' + str(index))

        import matplotlib.pyplot as plt
        
        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(before[0,:,:,index].max()), abs(before[0,:,:,index].min()))
        flatten_before = before[0,:,:,index].flatten()
        plt.pcolor(before[0,:,:,index], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_before.png')

        plt.close('all')
        plt.hist(flatten_before)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_before_hist.png')


        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(after[0,:,:,index].max()), abs(after[0,:,:,index].min()))
        flatten_after = after[0,:,:,index].flatten()
        plt.pcolor(after[0,:,:,index], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_after.png')

        plt.close('all')
        plt.hist(flatten_after)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_after_hist.png')
        
        

        #=======REVISION: Visualize the code information =======
        import matplotlib.pyplot as plt
        vis_z, z_likelihoods, vis_y, orig_mean, orig_variance, orig_prob, likelihoods, vis_x_hat, vis_x = \
               sess.run([z_tilde, z_likelihoods, y_hat, mean, variance, prob, y_likelihoods, x_hat, x])

        print(vis_y.max())
        print(vis_y.min())
        print(vis_y.shape)


        # Calculate the Ak    
        y_channel_wise_energy = np.var(vis_y[:, :,:,:], axis=(0,1,2)) 
        x_features_index = np.arange(0, args.num_filters, 1)
        Ak = y_channel_wise_energy / np.sum(y_channel_wise_energy)
        print(y_channel_wise_energy.shape)
        print(x_features_index.shape)
        plt.close('all')
        #ax = plt.gca()
        #ax.set_aspect(2/3)
        plt.bar(x_features_index, Ak)
        #plt.title("Ak (variance) for each channel")
        #plt.show()
        plt.savefig(args.output[:-4] + '_Ak.png')

        
        vis_variance = np.var(vis_y[0,:,:,:], axis=(0,1))        
        #print(vis_variance.max())
        #=========Find the channel with largest entropy============
        #index = np.argsort(vis_variance)[-3:]
        index = np.argmax(vis_variance)
        print('The channel with the largest entropy is ' + str(index))
        
        vis_inf = vis_y[0, :, :, index]
        vis_mean = orig_mean[0, :, :, index, :]
        vis_scale = orig_variance[0, :, :, index, :]
        vis_prob = orig_prob[0, :, :, index, :]

        #======Find the location for Distributions/Likelihoods Visualization=============


        import scipy.special
        TINY = 1e-10
        symbol = np.arange(start = vis_y.min(), stop= vis_y.max(), step = 1)
        #print(symbol)

        vis_locations = np.array([[5,30], [9,22], [28, 13], [19, 32]]) #[15,15], [17, 5], [26, 35], [31, 28], [31,7]])#, [21,5]
        #[loc_y, loc_x]


        
        for i in range(len(vis_locations)):
          loc_y, loc_x = vis_locations[i]
          
          likelihoods_0 = 0.5 * (1 + scipy.special.erf((symbol + 0.5 - vis_mean[loc_y, loc_x, 0]) / ((vis_scale[loc_y, loc_x, 0] + TINY) * 2 ** 0.5))) -\
                          0.5 * (1 + scipy.special.erf((symbol - 0.5 - vis_mean[loc_y, loc_x, 0]) / ((vis_scale[loc_y, loc_x, 0] + TINY) * 2 ** 0.5)))
          likelihoods_1 = 0.5 * (1 + scipy.special.erf((symbol + 0.5 - vis_mean[loc_y, loc_x, 1]) / ((vis_scale[loc_y, loc_x, 1] + TINY) * 2 ** 0.5))) -\
                          0.5 * (1 + scipy.special.erf((symbol - 0.5 - vis_mean[loc_y, loc_x, 1]) / ((vis_scale[loc_y, loc_x, 1] + TINY) * 2 ** 0.5)))
          likelihoods_2 = 0.5 * (1 + scipy.special.erf((symbol + 0.5 - vis_mean[loc_y, loc_x, 2]) / ((vis_scale[loc_y, loc_x, 2] + TINY) * 2 ** 0.5))) -\
                          0.5 * (1 + scipy.special.erf((symbol - 0.5 - vis_mean[loc_y, loc_x, 2]) / ((vis_scale[loc_y, loc_x, 2] + TINY) * 2 ** 0.5)))

          weighted_likelihoods = likelihoods_0*vis_prob[loc_y, loc_x, 0] + \
                                 likelihoods_1*vis_prob[loc_y, loc_x, 1] + \
                                 likelihoods_2*vis_prob[loc_y, loc_x, 2]
          print(np.sum(weighted_likelihoods))

          
          plt.close('all')
          ax = plt.gca()
          plt.bar(symbol, weighted_likelihoods)
          plt.xlabel('y_hat')
          plt.ylabel('likelihoods')
          string = '[ver=' + str(loc_y) + ', hor=' + str(loc_x) + '] \n y_hat: %.2f \n y_likelihoods: %.4f \n Mean: %.2f, %.2f, %.2f \n Scale: %.2f, %.2f, %.2f \n Weights:  %.2f, %.2f, %.2f' \
                   % (vis_inf[loc_y, loc_x],\
                      weighted_likelihoods[int(vis_inf[loc_y, loc_x] - vis_y.min())],\
                      vis_mean[loc_y, loc_x,0], vis_mean[loc_y, loc_x,1], vis_mean[loc_y, loc_x,2], \
                      vis_scale[loc_y, loc_x,0], vis_scale[loc_y, loc_x,1], vis_scale[loc_y, loc_x,2], \
                      vis_prob[loc_y, loc_x,0], vis_prob[loc_y, loc_x,1], vis_prob[loc_y, loc_x,2] )
          
          plt.text(0.01, 0.98, string, \
                   fontsize=12,\
                   horizontalalignment='left',\
                   verticalalignment='top',\
                   transform = ax.transAxes)

          plt.plot([vis_inf[loc_y, loc_x], vis_inf[loc_y, loc_x]], [0.0, weighted_likelihoods[int(vis_inf[loc_y, loc_x] - vis_y.min())]], 'r-', lw=2)
          
          plt.savefig(args.output[:-4] + '_location_[y=' + str(loc_y) + ', x=' + str(loc_x) + ']_likelihoods.png')

        #=================================================
        

        normalized_0 = (vis_inf - vis_mean[:,:,0])/vis_scale[:,:,0]
        normalized_1 = (vis_inf - vis_mean[:,:,1])/vis_scale[:,:,1]
        normalized_2 = (vis_inf - vis_mean[:,:,2])/vis_scale[:,:,2]

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        rg = max(abs(vis_inf.max()), abs(vis_inf.min()))
        plt.pcolor(vis_inf, cmap='RdBu', vmax=rg, vmin=-rg)
        #plt.pcolor(vis_inf, cmap='RdBu_r', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        
        #loc_x, loc_y = [31, 7]
        #for i in range(len(vis_locations)):
        #  loc_y, loc_x = vis_locations[i]
        #  plt.annotate('[' + str(loc_y) + ', ' + str(loc_x) + ']', xy = (loc_x, loc_y), xytext = (loc_x, loc_y - 2), \
        #               fontsize=13,\
        #               arrowprops=dict(arrowstyle="->", facecolor='black'))
        
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_y_hat.png')

        #====Mixtures of means and scales===========

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_mean.max()), abs(vis_mean.min()))
        plt.pcolor(vis_mean[:,:,0], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_mean_0.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_mean.max()), abs(vis_mean.min()))
        plt.pcolor(vis_mean[:,:,1], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_mean_1.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_mean.max()), abs(vis_mean.min()))
        plt.pcolor(vis_mean[:,:,2], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_mean_2.png')
        

        #====Mixtures of scales===========
        rg = 25

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.pcolor(vis_scale[:,:,0], cmap='Blues', vmax=rg, vmin=0)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_scale_0.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.pcolor(vis_scale[:,:,1], cmap='Blues', vmax=rg, vmin=0)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_scale_1.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.pcolor(vis_scale[:,:,2], cmap='Blues', vmax=rg, vmin=0)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_scale_2.png')

        # ======================
        rg = 10

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #rg = max(abs(normalized_0.max()), abs(normalized_0.min()))
        plt.pcolor(normalized_0, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_normalized_0.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #rg = max(abs(normalized_1.max()), abs(normalized_1.min()))
        plt.pcolor(normalized_1, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_normalized_1.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #rg = max(abs(normalized_2.max()), abs(normalized_2.min()))
        plt.pcolor(normalized_2, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_normalized_2.png')

        # ======================
        #====Mixtures of probs===========

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_prob.max()), abs(vis_prob.min()))
        plt.pcolor(vis_prob[:,:,0], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_prob_0.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_prob.max()), abs(vis_prob.min()))
        plt.pcolor(vis_prob[:,:,1], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_prob_1.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_prob.max()), abs(vis_prob.min()))
        plt.pcolor(vis_prob[:,:,2], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_prob_2.png')

        # ======================


        
        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.pcolor(-np.log2(likelihoods[0, :, :, index]), cmap='Reds', vmax=16, vmin=0)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_y_bits.png')

        # To check the z information
        z_variance = np.var(vis_z[0,:,:,:], axis=(0,1))
        print(z_variance.shape)
        z_index = np.argmax(z_variance)
        
        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rg = max(abs(vis_z.max()), abs(vis_z.min()))
        plt.pcolor(vis_z[0, :,:, z_index], cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(z_index) + '_z_hat.png')


        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.pcolor(-np.log2(z_likelihoods[0, :, :, z_index]), cmap='Reds')
        plt.colorbar(fraction=0.031, pad=0.04)
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(z_index) + '_z_bits.png')
        
        #----------------------------------------
        
        
        #To check the pixel shift for x
        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.pcolor(-np.log2(x_likelihoods[0, :, :, 0]), cmap='Reds')
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_x_bits_R.png')


        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.pcolor(-np.log2(x_likelihoods[0, :, :, 1]), cmap='Reds')
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_x_bits_G.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.pcolor(-np.log2(x_likelihoods[0, :, :, 2]), cmap='Reds')
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_x_bits_B.png')
        
        
        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        error = vis_x_hat - vis_x*255.0
        error = error[0, :, :, 0] #Only the R components
        rg = max(abs(error.max()), abs(error.min()))
        plt.pcolor(error, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_(x_hat-x)_R.png')        

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        error = vis_x_hat - vis_x*255.0
        error = error[0, :, :, 1] #Only the G components
        rg = max(abs(error.max()), abs(error.min()))
        plt.pcolor(error, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_(x_hat-x)_G.png')

        plt.close('all')
        ax = plt.gca()                           
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        error = vis_x_hat - vis_x*255.0
        error = error[0, :, :, 2] #Only the B components
        rg = max(abs(error.max()), abs(error.min()))
        plt.pcolor(error, cmap='RdBu', vmax=rg, vmin=-rg)
        plt.colorbar()
        plt.savefig(args.output[:-4] + '_largest_channel_' + str(index) + '_(x_hat-x)_B.png')
        '''



        

def decompress():
  """Decompresses an image."""

  # Set up for Range Coder
  range_coder_precision = 16
  

  with tf.device('/cpu:0'):
    # Read the shape information and compressed string from the binary file.
    with open(args.input, "rb") as f:
      x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
      y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
      length, minmax = np.frombuffer(f.read(4), dtype=np.uint16)
      string = f.read(length)
      string_y = f.read()


    y_shape = [int(s) for s in y_shape] + [args.num_filters]
    z_shape = [y_shape[0]//4] + [y_shape[1]//4] + [args.num_filters]


    # Add a batch dimension, then decompress and transform the image back.
    strings = tf.expand_dims(string, 0)

    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
    z_tilde = entropy_bottleneck.decompress(
        strings, z_shape, channels=args.num_filters)
    phi = hyper_synthesis(z_tilde, args.num_filters)

    # Transform the quantized image back (if requested).
    #Masked CNN by decoding one pixel by one pixel
    y = tf.placeholder(tf.float32, shape=(1, y_shape[0], y_shape[1], y_shape[2]))
    phi_fixed = tf.placeholder(tf.float32, shape=(1, y_shape[0], y_shape[1], args.num_filters*2))
    
    _, _, mean, variance = entropy_parameter(phi_fixed, y, args.num_filters, training=False)

    #---------------------------------

    #final_mean = tf.placeholder(tf.float32, shape=(1, y_shape[0], y_shape[1], y_shape[2]))
    #final_variance = tf.placeholder(tf.float32, shape=(1, y_shape[0], y_shape[1], y_shape[2]))
    
    # Sample the density up to `minmax` around the median.
    minmax = tf.convert_to_tensor(minmax, dtype=tf.float32)

    samples = math_ops.range(-minmax, minmax + 1, dtype=tf.float32)
    samples = tf.reshape(samples, [1, -1])
    
    half = constant_op.constant(.5, dtype=tf.float32)

    mean = tf.reshape(tf.layers.flatten(mean), [-1, 1])
    variance = tf.reshape(tf.layers.flatten(variance), [-1, 1])

    dist = tfd.Normal(loc = mean, scale = math_ops.abs(variance))    
    pmf = dist.cdf(samples+half) - dist.cdf(samples-half)

    # Add tail masses to first and last bin of pmf, as we clip values for
    # compression, meaning that out-of-range values get mapped to these bins.
    pmf = array_ops.concat([
      math_ops.add_n([pmf[:, :1], dist.cdf(-minmax-half)]),
      pmf[:, 1:-1],
      math_ops.add_n([pmf[:, -1:], 1 - dist.cdf(minmax+half)]),
    ], axis=-1)

    cdf = coder_ops.pmf_to_quantized_cdf(pmf, precision=range_coder_precision)
    cdf = array_ops.expand_dims(cdf, axis=1)

    string_y = tf.convert_to_tensor(string_y)
    
    decoded_values = tfc.range_decode(string_y, [y_shape[0]*y_shape[1]*args.num_filters, 1], cdf, precision=range_coder_precision)

    y_hat = tf.to_float(tf.reshape(decoded_values, [1, y_shape[0], y_shape[1], args.num_filters])) - minmax
    
    x_hat = synthesis_transform(y_hat, args.num_filters)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = save_image(args.output, x_hat)
       
    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
      latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
      tf.train.Saver().restore(sess, save_path=latest)
      
      phi_values = sess.run(phi)
      print(phi_values.shape)

      y_values = np.zeros((1, y_shape[0], y_shape[1], y_shape[2]), dtype='float32')
      #mean_values = np.zeros((1, y_shape[0], y_shape[1], y_shape[2]), dtype='float32')
      #variance_values = np.zeros((1, y_shape[0], y_shape[1], y_shape[2]), dtype='float32')

      for i in range(y_shape[0]):
        for j in range(y_shape[1]):
          #for k in range(y_shape[2]):

            start = time.time()
            next_y_values = sess.run(y_hat, feed_dict={y: y_values, phi_fixed: phi_values})
            y_values[:, i, j, :] = next_y_values[:, i, j, :]
            #mean, variance = sess.run([mean, variance], feed_dict={y: y_values, phi_fixed: phi_values})
            end = time.time()
            print("Time (s): {:0.3f}".format(end-start))
            #print(y_values[:, i, j, :])
                                              
      sess.run(op, feed_dict={y: y_values, phi_fixed: phi_values})
      
      

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train_joint_3_MSSSIM_128filter_3x3(4)_v1.0_subpixel_GMM_NLM",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--log_dir", default="log_dir",
      help="Directory where to save Tensorboard logs.")
  parser.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--valid_glob", default="valid_imgs/*.png",
      help="Glob pattern identifying validation data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--test_glob", default="test_imgs/*.png",
      help="Glob pattern identifying test data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.1, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--epochs", type=int, default=36,
      help="Number of epochs for training.")
  parser.add_argument(
      "--learning_rate", type=float, default=1e-4, dest="learning_rate",
      help="the learning rate for optimizer.")
  parser.add_argument(
      "--profiling_comp", type=bool, default=False,
      help="If True, create .json runtime profiling file of the compression.")

  args = parser.parse_args()

  # Create log directory for Tensorboard and overwrite existing run if it exists
  train_log_dir = os.path.join(args.log_dir, 'training/')
  val_log_dir = os.path.join(args.log_dir, 'validation/')
  if os.path.exists(os.path.dirname(args.log_dir)):
    for file in os.listdir(args.log_dir):
      file_path = os.path.join(args.log_dir, file)
      try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except Exception as e:
        print(e)
    else:
        os.makedirs(os.path.dirname(args.log_dir))
    os.makedirs(os.path.dirname(train_log_dir))
    os.makedirs(os.path.dirname(val_log_dir))

  if args.command == "train":
    train()
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress()
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress()
