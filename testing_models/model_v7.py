# V7 of Road classifier model
#   -Fully connected layer
#
# Architecture:
#   SECTION 1:
#		-Convolution - 5x5 kernel, 4 output channels
#
#   SECTION 2:
#   -Downsampling convolution - 5x5 kernel, 8 output channels
#   -Convolution - 5x5 kernel
#
#   SECTION 3:
#   -Downsampling convolution - 5x5 kernel, 16 output channels
#   -Convolution - 5x5 kernel
#
#   SECTION 4:
#   -Convolution - 5x5 kernel, 1 output channel
#   -Fully connected layer
#   -Convolution - 5x5 kernel, 16 output channels
#
#   SECTION 5:
#   -Upsampling convolution - 5x5 kernel
#   -Convolution - 5x5 kernel
#
#   SECTION 6:
#   -Upsampling convolution - 5x5 kernel
#   -Convolution - 5x5 kernel
#
#   SECTION 7:
#   -Convolution - 5x5 kernel, 1 output channel

import tensorflow as tf
import TensorflowUtils as utils
import scipy
import parameters as params

ch = [4, 8, 16, 16, 8, 4]  # Number of channels for each section (except output section)
k_size = 5;

# Dropout variable
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 1])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])
prev_y = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"]])

prev_y_in = tf.sub(tf.expand_dims(prev_y, 3), 0.5)
x_in = tf.concat(3, [x, prev_y_in])
ch_in = 2

##############################
# Section 1

# Convolution
layer_name = "s1_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch_in, ch[0]])
  b = utils.bias_variable([ch[0]])
  conv = utils.conv2d(x_in, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s1_conv1 = tf.nn.dropout(tanh, keep_prob)


##############################
# Section 2

# Downsampling convolution
layer_name = "s2_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[0], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s1_conv1, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s2_conv1 = tf.nn.dropout(tanh, keep_prob)

# Convolution
layer_name = "s2_conv2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 3

# Downsampling convolution
layer_name = "s3_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[1], ch[2]])
  b = utils.bias_variable([ch[2]])
  conv = utils.conv2d(s2_conv2, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s3_conv1 = tf.nn.dropout(tanh, keep_prob)

# Convolution
layer_name = "s3_conv2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[2], ch[2]])
  b = utils.bias_variable([ch[2]])
  conv = utils.conv2d(s3_conv1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s3_conv2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 4

# Convolution down to one channel
layer_name = "s4_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 1, ch[3], 1])
  b = utils.bias_variable([1])
  conv = utils.conv2d(s3_conv2, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s4_conv1 = tf.nn.dropout(tanh, keep_prob)

# Fully connected layer
layer_name = "s4_fc1"
with tf.name_scope(layer_name):
  # Get size of weights
  s4_shape = s4_conv1.get_shape()
  num_pixels_s4 = s4_shape[1].value*s4_shape[2].value

  W = utils.weight_variable([num_pixels_s4, num_pixels_s4])
  b = utils.bias_variable([num_pixels_s4])

  # Reshape, multiply by weights, add bias
  s4_conv1_flat = tf.reshape(s4_conv1, [-1, num_pixels_s4])
  fc = tf.add(tf.matmul(s4_conv1_flat, W), b)

  tanh = tf.nn.tanh(fc)
  s4_fc1 = tf.nn.dropout(tanh, keep_prob)

# Convolution up to original dimensionality (with skip connection from section 3)
layer_name = "s4_conv2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 1, 1, ch[3]])
  b = utils.bias_variable([ch[3]])

  s4_shape = s4_conv1.get_shape()
  s4_fc1_array = tf.reshape(s4_fc1, [-1, s4_shape[1].value, s4_shape[2].value, 1])

  conv = utils.conv2d(s4_fc1_array, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s4_conv2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 5

# Upsampling convolution (with skip connection from section 2)
layer_name = "s5_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[4], ch[3]])
  b = utils.bias_variable([ch[4]])

  conv = utils.conv2d_transpose(s4_conv2, W, b, tf.shape(s2_conv1), 2) 
  tanh = tf.nn.tanh(conv)
  s5_conv1 = tf.nn.dropout(tanh, keep_prob)

# Convolution
layer_name = "s5_conv2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[4], ch[4]])
  b = utils.bias_variable([ch[4]])
  conv = utils.conv2d(s5_conv1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s5_conv2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 6

# Upsampling convolution (with skip connection from section 1)
layer_name = "s6_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[5], ch[4]])
  b = utils.bias_variable([ch[5]])

  conv = utils.conv2d_transpose(s5_conv2, W, b, tf.shape(s1_conv1), 2)
  tanh = tf.nn.tanh(conv)
  s6_conv1 = tf.nn.dropout(tanh, keep_prob)

# Convolution
layer_name = "s6_conv2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[5], ch[5]])
  b = utils.bias_variable([ch[5]])
  conv = utils.conv2d(s6_conv1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s6_conv2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 7

# Convolution down to one channel
layer_name = "s7_conv1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([k_size, k_size, ch[5], 1])
  b = utils.bias_variable([1])
  conv = utils.conv2d(s6_conv2, W, b, 1)

  s7_conv1 = conv

##############################

# Form logits and a prediction
y = tf.squeeze(s7_conv1, squeeze_dims=[3])
prediction = tf.cast(tf.round(tf.sigmoid(y)), tf.bool)

# Image summarries
tf.image_summary("input", x, max_images=2)
tf.image_summary("label", tf.expand_dims(tf.cast(y_, tf.float32), dim=3), max_images=2)
tf.image_summary("prediction", tf.expand_dims(tf.cast(prediction, tf.float32), dim=3), max_images=2)
