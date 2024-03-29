
"""Simple model to regress 3d human poses from 2d joint locations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import cameras as cam

def kaiming(shape, dtype, partition_info=None):
  """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

  Args
    shape: dimensions of the tf array to initialize
    dtype: data type of the array
    partition_info: (Optional) info about how the variable is partitioned.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
      Needed to be used as an initializer.
  Returns
    Tensorflow array with initial weights
  """
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))


##################线性模型是一个类#######################
class LinearModel(object):
  """ A simple Linear+RELU model """

  def __init__(self, #指的是类实例本身
               linear_size,
               num_layers,
               residual,
               batch_norm,
               max_norm,
               batch_size,
               learning_rate,
               summaries_dir, #experiment/log
               predict_14=False,
               dtype=tf.float32):
    """Creates the linear + relu model

    Args
      linear_size: integer. number of units in each layer of the model
      num_layers: integer. number of bilinear blocks in the model
      residual: boolean. Whether to add residual connections
      batch_norm: boolean. Whether to use batch normalization
      max_norm: boolean. Whether to clip weights to a norm of 1
      batch_size: integer. The size of the batches used during training
      learning_rate: float. Learning rate to start with
      summaries_dir: String. Directory where to log progress
      predict_14: boolean. Whether to predict 14 instead of 17 joints
      dtype: the data type to use to store internal variables
    """

    # There are in total 17 joints in H3.6M and 16 in MPII (and therefore in stacked
    # hourglass detections). We settled with 16 joints in 2d just to make models
    # compatible (e.g. you can train on ground truth 2d and test on SH detections).  groundtruth2D，
    # This does not seem to have an effect on prediction performance.
    self.HUMAN_2D_SIZE = 16 * 2

    # In 3d all the predictions are zero-centered around the root (hip) joint, so
    # we actually predict only 16 joints. The error is still computed over 17 joints,
    # because if one uses, e.g. Procrustes alignment, there is still error in the
    # hip to account for!
    # There is also an option to predict only 14 joints, which makes our results
    # directly comparable to those in https://arxiv.org/pdf/1611.09010.pdf
    self.HUMAN_3D_SIZE = 14 * 3 if predict_14 else 16 * 3

    self.input_size  = self.HUMAN_2D_SIZE
    self.output_size = self.HUMAN_3D_SIZE

    self.isTraining = tf.placeholder(tf.bool,name="isTrainingflag")  #placeholder相当于定义了一个位置，这个位置的数据在程序运行的时候再指定，这样程序中就可以不用定义大量的常量来提供输入数据。因为相同的常量在多次迭代中不希望重复定义
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter( os.path.join(summaries_dir, 'train' ))#汇总结果写入事件（event file）写到experiment/log/train
    self.test_writer  = tf.summary.FileWriter( os.path.join(summaries_dir, 'test' ))

    self.linear_size   = linear_size#每一层的unit数
    self.batch_size    = batch_size#batchsize的大小
    #*********************这里一套exponential_decay都是按经验给的************************
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")#不可训练
    self.global_step   = tf.Variable(0, trainable=False, name="global_step") # to compute the decayed learning rate. 
    decay_steps = 100000  # empirical
    decay_rate = 0.96     # empirical
    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)

    # === Transform the inputs ===##通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生成一个上下文管理器并明确指定tf.get_variable将直接获取已生成变量
    with vs.variable_scope("inputs"): #在“inputs”命名空间内创建名字为“enc_in"和”dec_out“的常量

      # in=2d poses, out=3d poses
      enc_in  = tf.placeholder(dtype, shape=[None, self.input_size], name="enc_in")  #编码器输入
      dec_out = tf.placeholder(dtype, shape=[None, self.output_size], name="dec_out")#解码器输出

      self.encoder_inputs  = enc_in
      self.decoder_outputs = dec_out

    # === Create the linear + relu combos ===
    with vs.variable_scope( "linear_model" ):    #with是一种上下文管理协议，简化try….except….finlally的处理流程

      # === First layer, brings dimensionality up to linear_size ===（这个就是在论文图中没显示的第一个linear layer）
      w1 = tf.get_variable( name="w1", initializer=kaiming, shape=[self.HUMAN_2D_SIZE, linear_size], dtype=dtype )
      b1 = tf.get_variable( name="b1", initializer=kaiming, shape=[linear_size], dtype=dtype )
      w1 = tf.clip_by_norm(w1,1) if max_norm else w1   #click by norm 是一种常用的防止梯度爆炸的方法
      y3 = tf.matmul( enc_in, w1 ) + b1

      if batch_norm:
        y3 = tf.layers.batch_normalization(y3,training=self.isTraining, name="batch_normalization")#对隐藏层的输入进行归一化处理
      y3 = tf.nn.relu( y3 )
      y3 = tf.nn.dropout( y3, self.dropout_keep_prob )

      # === Create multiple bi-linear layers ===创建2个双线性层   num_layers: integer. number of bilinear blocks in the model
      for idx in range( num_layers ):#idx为0到num_layer-1  
        y3 = self.two_linear( y3, linear_size, residual, self.dropout_keep_prob, max_norm, batch_norm, dtype, idx )  #给线性层增加residual connection（问题：idx的for循环的作用）

      # === Last linear layer has HUMAN_3D_SIZE in output ===
      w4 = tf.get_variable( name="w4", initializer=kaiming, shape=[linear_size, self.HUMAN_3D_SIZE], dtype=dtype )
      b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[self.HUMAN_3D_SIZE], dtype=dtype )
      w4 = tf.clip_by_norm(w4,1) if max_norm else w4
      y = tf.matmul(y3, w4) + b4
      # === End linear model ===

    # Store the outputs here
    self.outputs = y
    self.loss = tf.reduce_mean(tf.square(y - dec_out))
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)  #用来显示标量信息的

    # To keep track of the loss in mm
    self.err_mm = tf.placeholder( tf.float32, name="error_mm" )
    self.err_mm_summary = tf.summary.scalar( "loss/error_mm", self.err_mm )

    # Gradients and update operation for training the model.
    opt = tf.train.AdamOptimizer( self.learning_rate )  #定义了反向传播的优化算法，比较常用的有Adam,Gradient和Momentum,只要运行了sess.run就可以对GraphKeys.TRAINABLE_VARIABLES中的变量进行优化
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #tf.GraphKeys.UPDATE_OPS其中保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
    
#该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行
    with tf.control_dependencies(update_ops):

      # Update all the trainable parameters
      gradients = opt.compute_gradients(self.loss)   #minimize()方法是compute_gradients和apply_gradients方法的结合
      self.gradients = [[] if i==None else i for i in gradients]
      self.updates = opt.apply_gradients(gradients, global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # To save the model      #tf.train.Saver的作用是save and restores variables  max_to_keep指的是保存模型的个数  且只保存tf.global_variables()中的变量 
    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10 )  


  def two_linear( self, xin, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, dtype, idx ):
    """
    Make a bi-linear block with optional residual connection

    Args
      xin: the batch that enters the block
      linear_size: integer. The size of the linear units
      residual: boolean. Whether to add a residual connection
      dropout_keep_prob: float [0,1]. Probability of dropping something out
      max_norm: boolean. Whether to clip weights to 1-norm
      batch_norm: boolean. Whether to do batch normalization
      dtype: type of the weigths. Usually tf.float32
      idx: integer. Number of layer (for naming/scoping)
    Returns
      y: the batch after it leaves the block
    """

    with vs.variable_scope( "two_linear_"+str(idx) ) as scope:

      input_size = int(xin.get_shape()[1])  #就得到了xin的列数

      # Linear 1
      w2 = tf.get_variable( name="w2_"+str(idx), initializer=kaiming, shape=[input_size, linear_size], dtype=dtype)
      b2 = tf.get_variable( name="b2_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w2 = tf.clip_by_norm(w2,1) if max_norm else w2
      y = tf.matmul(xin, w2) + b2
      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization1"+str(idx))

      y = tf.nn.relu( y )
      y = tf.nn.dropout( y, dropout_keep_prob )

      # Linear 2
      w3 = tf.get_variable( name="w3_"+str(idx), initializer=kaiming, shape=[linear_size, linear_size], dtype=dtype)
      b3 = tf.get_variable( name="b3_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w3 = tf.clip_by_norm(w3,1) if max_norm else w3
      y = tf.matmul(y, w3) + b3

      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization2"+str(idx))

      y = tf.nn.relu( y )
      y = tf.nn.dropout( y, dropout_keep_prob )

      # Residual every 2 blocks
      y = (xin + y) if residual else y

    return y

  def step(self, session, encoder_inputs, decoder_outputs, dropout_keep_prob, isTraining=True):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use
      encoder_inputs: list of numpy vectors to feed as encoder inputs
      decoder_outputs: list of numpy vectors that are the expected decoder outputs
      dropout_keep_prob: (0,1] dropout keep probability
      isTraining: whether to do the backward step or only forward

    Returns
      if isTraining is True, a 4-tuple
        loss: the computed loss of this batch
        loss_summary: tf summary of this batch loss, to log on tensorboard
        learning_rate_summary: tf summary of learnign rate to log on tensorboard
        outputs: predicted 3d poses
      if isTraining is False, a 3-tuple
        (loss, loss_summary, outputs) same as above
    """

    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.isTraining: isTraining,
                  self.dropout_keep_prob: dropout_keep_prob}

    # Output feed: depends on whether we do a backward step or not.
    if isTraining:
      output_feed = [self.updates,       # Update Op that does SGD
                     self.loss,
                     self.loss_summary,
                     self.learning_rate_summary,
                     self.outputs]

      outputs = session.run( output_feed, input_feed )
      return outputs[1], outputs[2], outputs[3], outputs[4]

    else:
      output_feed = [self.loss, # Loss for this batch.
                     self.loss_summary,
                     self.outputs]

      outputs = session.run(output_feed, input_feed)
      return outputs[0], outputs[1], outputs[2]  # No gradient norm

  def get_all_batches( self, data_x, data_y, camera_frame, training=True ):
    """
    Obtain a list of all the batches, randomly permutted
    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      training: True if this is a training batch. False otherwise.

    Returns
      encoder_inputs: list of 2d batches
      decoder_outputs: list of 3d batches
    """

    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d
    #print("getbatch的维度为多少"，n)
    encoder_inputs  = np.zeros((n, self.input_size), dtype=float) #n*32
    decoder_outputs = np.zeros((n, self.output_size), dtype=float) #n*48

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      #print("想知道这个2d和3d数据的大小",data_x[ key2d ].shape,data_y[ key3d ].shape,idx,n2d,n)  #(1612*32)和（1612*48)就有了
      #assert 1==2,"debug结束"
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d


    if training:
      # Randomly permute everything
      idx = np.random.permutation( n )
      encoder_inputs  = encoder_inputs[idx, :]
      decoder_outputs = decoder_outputs[idx, :]
      #print(encoder_inputs.shape) 在这的时候，数据的大小还是n*32和n*48

    # Make the number of examples a multiple of the batch size
    n_extra  = n % self.batch_size #n_extra=8
    if n_extra > 0:  # Otherwise examples are already a multiple of batch size
      encoder_inputs  = encoder_inputs[:-n_extra, :]
      decoder_outputs = decoder_outputs[:-n_extra, :]

    #按照batch的大小对输入和输出进行切片，切成[array(n,32/48)...,...]
    n_batches = n // self.batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )

    return encoder_inputs, decoder_outputs
