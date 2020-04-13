#!/usr/bin/env python3


import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow_compression as tfc
import math
import time
import scipy.special

from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
  
  


def residualblock(tensor, num_filters, scope="residual_block"):
  """Builds the residual block"""
  with tf.variable_scope(scope):
    with tf.variable_scope("conv0"):
      layer = tfc.SignalConv2D(
        num_filters//2, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(tensor)

    with tf.variable_scope("conv1"):
      layer = tfc.SignalConv2D(
        num_filters//2, (3,3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(output)

    with tf.variable_scope("conv2"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      output = layer(output)
      
    tensor = tensor + output
       
  return tensor


def NonLocalAttentionBlock(input_x, num_filters, scope="NonLocalAttentionBlock"):
  """Builds the non-local attention block"""
  with tf.variable_scope(scope):
    trunk_branch = residualblock(input_x, num_filters, scope="trunk_RB_0")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_1")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_2")
    
    
    attention_branch = residualblock(input_x, num_filters, scope="attention_RB_0")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_1")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_2")

    with tf.variable_scope("conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
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
            use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor2 = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_1"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor2 = layer(tensor2)
        
        tensor = tensor + tensor2


      if i < 3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):
          shortcut = tfc.SignalConv2D(num_filters, (1, 1), corr=True, strides_down=2, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

        if i == 1:
          #Add one NLAM
          tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")
          

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
            use_bias=False, activation=None, name='signal_conv2d') 
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
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters     
    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters 
    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
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
          use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        tensor2 = layer(tensor)

      with tf.variable_scope("Block_" + str(i) + "_layer_1"):
        layer = tfc.SignalConv2D(
          num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        tensor2 = layer(tensor2)
        tensor = tensor + tensor2


      if i <3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):

          # Use Sub-Pixel to replace deconv.
          shortcut = tfc.SignalConv2D(num_filters*4, (1, 1), corr=False, strides_up=1, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)
          shortcut_tensor = tf.depth_to_space(shortcut_tensor, 2)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):

          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(num_filters*4, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)         
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='igdn', inverse=True), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          
          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(12, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=None, name='signal_conv2d')
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
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters*2, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
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
      noise = tf.random_uniform(tf.shape(inputs), -half, half)
      values = tf.add_n([inputs, noise])

    else: #inference
      #if inputs is not None: #compress
      values = tf.round(inputs)
        

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


    #=========Gaussian Mixture Model=========
    prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
             tf.split(tensor, num_or_size_splits=9, axis = 3)
    scale0 = tf.abs(scale0)
    scale1 = tf.abs(scale1)
    scale2 = tf.abs(scale2)



    probs = tf.stack([prob0, prob1, prob2], axis=-1)
    probs = tf.nn.softmax(probs, axis=-1)
  
    # To merge them together
    means = tf.stack([mean0, mean1, mean2], axis=-1)
    variances = tf.stack([scale0, scale1, scale2], axis=-1)

    # =======================================
    '''
    #Calculate the likelihoods for inputs
    #if inputs is not None:
    if training:

      dist_0 = tfd.Normal(loc = mean0, scale = scale0, name='dist_0')
      dist_1 = tfd.Normal(loc = mean1, scale = scale1, name='dist_1')
      dist_2 = tfd.Normal(loc = mean2, scale = scale2, name='dist_2')

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
      #values = None
      likelihoods = None

    '''
    likelihoods = None
        
  return values, likelihoods, means, variances, probs





def compress(input, output, num_filters, checkpoint_dir):
  
  start = time.time()
  tf.set_random_seed(1)
  tf.reset_default_graph()
  
  
  with tf.device('/cpu:0'):
    # Load input image and add batch dimension.
    
    x = load_image(input)

    # Pad the x to x_pad
    mod = tf.constant([64, 64, 1], dtype=tf.int32)
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
    y = analysis_transform(x_pad, num_filters)

    # Build a hyper autoencoder
    z = hyper_analysis(y, num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    string = entropy_bottleneck.compress(z)
    string = tf.squeeze(string, axis=0)

    z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

    # To decompress the z_tilde back to avoid the inconsistence error
    string_rec = tf.expand_dims(string, 0)
    z_tilde = entropy_bottleneck.decompress(string_rec, tf.shape(z)[1:], channels=num_filters)

    phi = hyper_synthesis(z_tilde, num_filters)


    # REVISION： for Gaussian Mixture Model (GMM), use window-based fast implementation    
    #y = tf.clip_by_value(y, -255, 256)
    y_hat = tf.round(y)


    tiny_y = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters])
    tiny_phi = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters*2]) 
    _, _, y_means, y_variances, y_probs = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)

    x_hat = synthesis_transform(y_hat, num_filters)


    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    x_hat = x_hat[0, :tf.shape(x)[1], :tf.shape(x)[2], :]

    #op = save_image('temp/temp.png', x_hat)
    
    # Mean squared error across pixels.
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    mse = tf.reduce_mean(tf.squared_difference(x * 255, x_hat))


    with tf.Session() as sess:
      #print(tf.trainable_variables())
      sess.run(tf.global_variables_initializer())
      # Load the latest model checkpoint, get the compressed string and the tensor
      # shapes.
      #latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
      
      latest = "models/model-1399000" #lambda = 14
        
      print(latest)
      tf.train.Saver().restore(sess, save_path=latest)

      
      
      string, x_shape, y_shape, num_pixels, y_hat_value, phi_value = \
              sess.run([string, tf.shape(x), tf.shape(y), num_pixels, y_hat, phi])
      

      
      minmax = np.maximum(abs(y_hat_value.max()), abs(y_hat_value.min()))
      minmax = int(np.maximum(minmax, 1))
      #num_symbols = int(2 * minmax + 3)
      print(minmax)
      #print(num_symbols)
      
      # Fast implementations by only encoding non-zero channels with 128/8 = 16bytes overhead
      flag = np.zeros(y_shape[3], dtype=np.int)
      
      for ch_idx in range(y_shape[3]):
        if np.sum(abs(y_hat_value[:, :,:, ch_idx])) > 0:
          flag[ch_idx] = 1

      non_zero_idx = np.squeeze(np.where(flag == 1))

                 
      num = np.packbits(np.reshape(flag, [8, y_shape[3]//8]))
           
      # ============== encode the bits for z===========
      if os.path.exists(output):
        os.remove(output)

      fileobj = open(output, mode='wb')
      fileobj.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      fileobj.write(np.array([len(string), minmax], dtype=np.uint16).tobytes())
      fileobj.write(np.array(num, dtype=np.uint8).tobytes())
      fileobj.write(string)
      fileobj.close()



      # ============ encode the bits for y ==========
      print("INFO: start encoding y")
      encoder = RangeEncoder(output[:-4] + '.bin')
      samples = np.arange(0, minmax*2+1)
      TINY = 1e-10

       

      kernel_size = 5
      pad_size = (kernel_size - 1)//2
      
      
      
      padded_y = np.pad(y_hat_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))
      padded_phi = np.pad(phi_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))

      
      for h_idx in range(y_shape[1]):
        for w_idx in range(y_shape[2]):          

          
          extracted_y = padded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]
          extracted_phi = padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]

          
          y_means_values, y_variances_values, y_probs_values = \
                          sess.run([y_means, y_variances, y_probs], \
                                   feed_dict={tiny_y: extracted_y, tiny_phi: extracted_phi})         

          
          
          for i in range(len(non_zero_idx)):
            ch_idx = non_zero_idx[i]
            
            mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]

            start00 = time.time()

            # Calculate the pmf/cdf            
            pmf = (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] +\
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]

            '''
            # Add the tail mass
            pmf[0] += 0.5 * (1 + scipy.special.erf(( -0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) * weight[0] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) * weight[1] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) * weight[2]
                      
            pmf[-1] += (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]
            '''
            
            # To avoid the zero-probability            
            pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
            pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
            cdf = list(np.add.accumulate(pmf_clip))
            cdf = [0] + [int(i) for i in cdf]
                      
            symbol = np.int(y_hat_value[0, h_idx, w_idx, ch_idx] + minmax )
            encoder.encode([symbol], cdf)


            

      encoder.close()

      size_real = os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin')
      
      bpp_real = (os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin'))* 8 / num_pixels
      bpp_side = (os.path.getsize(output))* 8 / num_pixels
      

      end = time.time()
      print("Time : {:0.3f}".format(end-start))

      psnr = sess.run(tf.image.psnr(x_hat, x*255, 255))
      msssim = sess.run(tf.image.ssim_multiscale(x_hat, x*255, 255))
      
      print("Actual bits per pixel for this image: {:0.4}".format(bpp_real))
      print("Side bits per pixel for z: {:0.4}".format(bpp_side))
      print("PSNR (dB) : {:0.4}".format(psnr[0]))
      print("MS-SSIM : {:0.4}".format(msssim[0]))


def decompress(input, output, num_filters, checkpoint_dir):
  """Decompresses an image by a fast implementation."""



  start = time.time()


  tf.set_random_seed(1)
  tf.reset_default_graph()

  with tf.device('/cpu:0'):

    print(input)
    
    # Read the shape information and compressed string from the binary file.
    fileobj = open(input, mode='rb')
    x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)
    length, minmax = np.frombuffer(fileobj.read(4), dtype=np.uint16)
    num = np.frombuffer(fileobj.read(16), dtype=np.uint8)
    string = fileobj.read(length)

    fileobj.close()

    flag = np.unpackbits(num)
    non_zero_idx = np.squeeze(np.where(flag == 1))


    # Get x_pad_shape, y_shape, z_shape
    pad_size = 64
    x_pad_shape = [1] + [int(math.ceil(x_shape[0] / pad_size) * pad_size)] + [int(math.ceil(x_shape[1] / pad_size) * pad_size)]  + [3]
    y_shape = [1] + [x_pad_shape[1]//16] + [x_pad_shape[2]//16] + [num_filters]
    z_shape = [y_shape[1]//4] + [y_shape[2]//4] + [num_filters]


    
 
    # Add a batch dimension, then decompress and transform the image back.
    strings = tf.expand_dims(string, 0)

    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
    z_tilde = entropy_bottleneck.decompress(
        strings, z_shape, channels=num_filters)
    phi = hyper_synthesis(z_tilde, num_filters)

    # Transform the quantized image back (if requested).
    tiny_y = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters])
    tiny_phi = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters*2])
    _, _, means, variances, probs = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)

    # Decode the x_hat usign the decoded y
    y_hat = tf.placeholder(dtype=tf.float32, shape=y_shape)
    x_hat = synthesis_transform(y_hat, num_filters)


    # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
    x_hat = x_hat[0, :int(x_shape[0]), :int(x_shape[1]), :]

    # Write reconstructed image out as a PNG file.
    op = save_image(output, x_hat)
       
    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
      #latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
      
      
      latest = "models/model-1399000" #lambda = 14
      print(latest)

        
      tf.train.Saver().restore(sess, save_path=latest)
      
      phi_value = sess.run(phi)

      print("INFO: start decoding y")
      print(time.time() - start)


      decoder = RangeDecoder(input[:-4] + '.bin')
      samples = np.arange(0, minmax*2+1)
      TINY = 1e-10

      
      # Fast implementation to decode the y_hat
      kernel_size = 5
      pad_size = (kernel_size - 1)//2

      decoded_y = np.zeros([1] + [y_shape[1]+kernel_size-1] + [y_shape[2]+kernel_size-1] + [num_filters])
      padded_phi = np.pad(phi_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))
      


      for h_idx in range(y_shape[1]):
        for w_idx in range(y_shape[2]):



          y_means, y_variances, y_probs = \
                   sess.run([means, variances, probs], \
                            feed_dict={tiny_y: decoded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :], \
                                       tiny_phi: padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]})
         

              
          for i in range(len(non_zero_idx)):
            ch_idx = non_zero_idx[i]

              
            mu = y_means[0, pad_size, pad_size, ch_idx, :] + minmax
            sigma = y_variances[0, pad_size, pad_size, ch_idx, :]
            weight = y_probs[0, pad_size, pad_size, ch_idx, :]


            pmf = (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] +\
                  (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) - \
                   0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]

            pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
            pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
            cdf = list(np.add.accumulate(pmf_clip))
            cdf = [0] + [int(i) for i in cdf]

            decoded_y[0, h_idx+pad_size, w_idx+pad_size, ch_idx] = decoder.decode(1, cdf)[0] - minmax 


      decoded_y = decoded_y[:, pad_size:y_shape[1]+pad_size, pad_size:y_shape[2]+pad_size, :]
                                                          
      sess.run(op, feed_dict={y_hat: decoded_y})
      
      end = time.time()
      print("Time (s): {:0.3f}".format(end-start))


