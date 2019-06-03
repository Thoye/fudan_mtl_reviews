import tensorflow as tf
from src.models.base_model import *

FLAGS = tf.app.flags.FLAGS

TASK_NUM=14

class MTLModel(BaseModel):

  def __init__(self, word_embed, all_data, adv, is_train):
    # input data
    # self.all_data = all_data
    self.is_train = is_train
    self.adv = adv

    # embedding initialization
    self.word_dim = word_embed.shape[1]
    w_trainable = True if self.word_dim==50 else False
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    
    self.shared_conv = ConvLayer('conv_shared', FILTER_SIZES)
    self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)  # 'linear_shared'是多分类

    self.tensors = []

    for task_name, data in all_data:
      with tf.name_scope(task_name):  # task_name: task_m, task_n
        self.build_task_graph(data)

  def adversarial_loss(self, feature, task_label):  # loss_adv
    '''make the task classifier cannot reliably predict the task based on 
    the shared feature   :使分类器无法区分
    '''
    # input = tf.stop_gradient(input)
    feature = flip_gradient(feature)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to TASK_NUM classes
    logits, loss_adv_l2 = self.shared_linear(feature)  # loss_adv_l2 是'linear_shared'层的w和b 的loss; 原来是多分类

    label = tf.one_hot(task_label, TASK_NUM)  # 多分类
    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    return loss_adv, loss_adv_l2  # loss_adv 是多分类的交叉熵; loss_adv_l2是'linear_shared'层的w和b 的loss
  
  def diff_loss(self, shared_feat, task_feat):  # 正交化约束
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)  # 私有层 conv_out
    shared_feat -= tf.reduce_mean(shared_feat, 0)  # 共享层 shared_out

    task_feat = tf.nn.l2_normalize(task_feat, 1)  # 不重要
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)  # 不重要

    correlation_matrix = tf.matmul(  # matmul 乘法
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01  # cost: 平方再求平均
    cost = tf.where(cost > 0, cost, 0, name='value')  # cost = max(0,cost)

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
      loss_diff = tf.identity(cost)

    return loss_diff  # conv_out, shared_out 先乘后平方再求平均, 即为正交化约束loss

  def build_task_graph(self, data):
    task_label, labels, sentence = data
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)  # tf.nn.embedding_lookup得到句子中每个word的embedding

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_task', FILTER_SIZES)  # 'conv_task'是个layer_name
    conv_out = conv_layer(sentence)  # 调用ConvLayer.call 返回一个conv_outs=[conv1,conv2,...]
    conv_out = max_pool(conv_out, 500)  # conv_out 私有层 []

    shared_out = self.shared_conv(sentence)  # self.shared_conv = ConvLayer('conv_shared', FILTER_SIZES)
    shared_out = max_pool(shared_out, 500)  # shared_out 共享层 []

    if self.adv:
      feature = tf.concat([conv_out, shared_out], axis=1)  # 加对抗 feature = concat([私有层 + 共享层])
    else:
      feature = conv_out  # 不加对抗

    if self.is_train:  # 无所谓，加上dropout
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    linear = LinearLayer('linear', 2, True)  # 'linear'是个layer_name; 原来是二分类
    logits, loss_l2 = linear(feature)  # logits = o = xw_plus_b 全连接; loss_l2 是'linear'层的w和b 的loss; 二分类
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, 2),  # labels 从 data 来的
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)  # loss_ce: loss_task 交叉熵loss

    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)  # loss_adv; (loss_adv_l2:'linear_shared'层的)
    loss_diff = self.diff_loss(shared_out, conv_out)  # loss_diff 正交化约束loss

    if self.adv:  # 加对抗需要加 loss_adv; (loss_adv_l2:w和b 的loss); loss_diff
      loss = loss_ce + 0.05*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2) + loss_diff
    else:  # 不加对抗只需要加 loss_ce; (loss_l2:w和b 的loss)
      loss = loss_ce  + FLAGS.l2_coef*loss_l2
    
    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors.append((acc, loss))
    # self.tensors.append((acc, loss, pred))

# ---------------------以上是重要部分-------------------------------------------------------------------------------
# ---------------------以下不重要----------------------------------------------------------------------------------

  def build_train_op(self):
    if self.is_train:
      self.train_ops = []
      for _, loss in self.tensors:
        train_op = optimize(loss)
        self.train_ops.append(train_op)

def build_train_valid_model(model_name, word_embed, all_train, all_test, adv, test):
  with tf.name_scope("Train"):
    with tf.variable_scope(model_name, reuse=None):
      m_train = MTLModel(word_embed, all_train, adv, is_train=True)
      m_train.set_saver(model_name)
      if not test:
        m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope(model_name, reuse=True):
      m_valid = MTLModel(word_embed, all_test, adv, is_train=False)
      m_valid.set_saver(model_name)
  
  return m_train, m_valid
