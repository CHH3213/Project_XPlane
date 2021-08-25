# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()  # 可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
# x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
# y_pred=tf.layers.dense(x,units=1)
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(y_pred))
# loss =tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
# print(sess.run(loss))
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train=optimizer.minimize(loss)
# count=0
# while True:
#     _,loss_value=sess.run((train,loss))
#     print(loss_value)
#     count+=1
#     if loss_value<0.00001:
#         print('count',count)
#         break


# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()  # 可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
# #simple demo
# # 定义一个计算图，实现两个向量的加法
# # 定义两个输入，a为常量，b为随机值
# a=tf.constant([10.0, 20.0, 40.0], name='a')
# b=tf.Variable(tf.random.uniform([3]), name='b')   # 从均匀分布中输出随机值,[3]代表张量尺寸
# output=tf.add_n([a,b], name='add')    #Add all input tensors element wise
#
# with tf.Session() as sess:
#     # 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日
#     writer=tf.summary.FileWriter('D:\\tf_dir\\tensorboard_study', sess.graph)
#     sess.run(tf.global_variables_initializer())
#     f=sess.run(output)
#     print(f)
#     writer.close()
#     pass

import torch    # 如正常则静默
a = torch.Tensor([1.])    # 如正常则静默
print(a)
a.cuda()    # 如正常则返回"tensor([ 1.], device='cuda:0')"
from torch.backends import cudnn # 如正常则静默
print(cudnn.is_acceptable(a.cuda()))    # 如正常则返回 "True"

import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())