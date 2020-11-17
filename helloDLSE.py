# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 15:55


import tensorflow as tf

hello = tf.constant("Hello DLSE: Deep Learning and Symbolic Executor for complex question answering")

sess = tf.Session()

print(sess.run(hello))