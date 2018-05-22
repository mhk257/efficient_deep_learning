
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



def cnn_block(data, n_filter, is_training, scope, scope2, reuse=False):
  with tf.variable_scope(scope):

    conv = tf.contrib.layers.conv2d(data, n_filter, [5, 5], activation_fn=None, padding='VALID',
            weights_initializer=tf.random_normal_initializer(stddev=0.01), biases_initializer=tf.zeros_initializer(), scope='conv', reuse=reuse)
    #conv = slim.layers.conv2d(data, n_filter, [5, 5], activation_fn=None, padding='VALID',
    #        weights_initializer=tf.random_normal_initializer(stddev=0.01), scope=scope, reuse=reuse)
    #norm = slim.layers.batch_norm(conv, scale=False, decay=0.9, epsilon=0.001, scope=scope2, is_training=is_training, reuse=reuse)

    norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, scope='bn', is_training=is_training, reuse=reuse)
  #reuse = tf.cast(reuse, tf.bool)

    relu = tf.nn.relu(norm)

  return relu

def cnn_block_wrelu(data, n_filter, is_training, scope, scope2, reuse=False):
  with tf.variable_scope(scope) as scope:
    conv = tf.contrib.layers.conv2d(data, n_filter, [5, 5], activation_fn=None, padding='VALID',
      weights_initializer=tf.random_normal_initializer(stddev=0.01), biases_initializer=tf.zeros_initializer(), scope='conv', reuse=reuse)

    #conv = slim.layers.conv2d(data, n_filter, [5, 5], activation_fn=None, padding='VALID',
    #        weights_initializer=tf.random_normal_initializer(stddev=0.01), scope=scope, reuse=reuse)
    #norm = slim.layers.batch_norm(conv, scale=False, decay=0.9, epsilon=0.001, scope=scope2, is_training=is_training, reuse=reuse)


    norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, scope='bn', is_training=is_training, reuse=reuse)

  return norm



def sia_eff_net(x_input, is_training, reuse=False):

  print(is_training)
  with tf.name_scope("model"):
    layer1_out = cnn_block(x_input, 32, is_training, "l1", "l1_bn", reuse)
    #print('Layer1 {}'.format(layer1_out.get_shape()))
    layer2_out = cnn_block(layer1_out, 32, is_training, "l2", "l2_bn", reuse)
    layer3_out = cnn_block(layer2_out, 64, is_training, "l3", "l3_bn", reuse)
    layer4_out = cnn_block(layer3_out, 64, is_training, "l4", "l4_bn", reuse)
    layer5_out = cnn_block(layer4_out, 64, is_training, "l5", "l5_bn", reuse)
    layer6_out = cnn_block(layer5_out, 64, is_training, "l6", "l6_bn", reuse)
    layer7_out = cnn_block(layer6_out, 64, is_training, "l7", "l7_bn", reuse)
    layer8_out = cnn_block(layer7_out, 64, is_training, "l8", "l8_bn", reuse)

    layer9_out = cnn_block_wrelu(layer8_out, 64, is_training, "l9", "l9_bn", reuse)

  return layer9_out

def sia_net_slim(inputs, is_training, scope="win37_dep9", reuse=False):
	num_maps = 64
	kw = 5
	kh = 5

	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], padding='VALID', reuse=reuse, activation_fn=tf.nn.relu,
		normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training},
		weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN',uniform=True, seed=None, dtype=tf.float32)) as scope:

			net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
			net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
			net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')

			net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None, normalizer_fn=None)
			net = slim.batch_norm(net, is_training=is_training)

	return net


def eff_stereo_loss(_inputs, _targets, batch_size, half_range):
  #out_val = tf.Variable(0, tf.float32)
  half_width = tf.constant(2, tf.float32)
  wt_val = tf.constant([0.05, 0.2, 0.5, 0.2, 0.05])
  #i = tf.constant(0)
  #while_condition = lambda i: tf.less(i, batch_size)

  idx_val = (tf.constant(0), tf.constant(0.0))

  def while_condition(i, out_val):
    return tf.less(i, batch_size)



  def body(i, out_val):

    s = tf.cast(_targets[i] - half_width, tf.int32)
    e = tf.cast(_targets[i] + half_width, tf.int32)

    probs =_inputs[i,s:e]

    #probs = tf.Print(probs, [probs])

    start = tf.cast(half_width + 1.0 - (_targets[i]-tf.cast(s, tf.float32)), tf.int32) - 1

    finish = tf.cast(half_width + 1.0 + (tf.cast(e, tf.float32)-_targets[i]), tf.int32) - 1

    wts = wt_val[start:finish]

    #wts = tf.Print(wts, [wts])
    prod = tf.reduce_sum(tf.multiply(probs, wts))



    out_val = out_val - prod

    return tf.add(i, 1), out_val


  return tf.while_loop(while_condition, body, idx_val)[1] / batch_size

def eff_stereo_loss2(_logits, _targets, batch_size, half_range):

  half_width = 2
  fixed_target = half_range + 1

  s = fixed_target - half_width
  e = fixed_target + half_width
  wt_val = np.array([0.05, 0.2, 0.5, 0.2, 0.05])

  _labels =  np.zeros((batch_size, half_range*2 + 1))

  start = half_width + 1 - (fixed_target - s) - 1

  finish = half_width + 1 + (e -fixed_target)

  _labels[:,s-1:e] = wt_val

  _labels = tf.constant(_labels, tf.float32)


  #softmax = tf.nn.softmax(_logits)



  #cross_entropy = -tf.reduce_sum(_labels * tf.log(softmax), reduction_indices=[1])

  #return tf.reduce_mean(cross_entropy, name='xentropy_mean')

  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=_labels, logits=_logits))

  #tf.clip_by_value(softmax,1e-10,1.0)
