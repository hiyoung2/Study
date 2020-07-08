# 당뇨병
# 회귀 모델

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_diabetes


diabetes = load_diabetes()
x_data = diabetes['data']
y_data = diabetes['target']

print("x_data.shape :", x_data.shape) # x_data.shape : (442, 10)
print("y_data.shape :", y_data.shape) # y_data.shape : (442,)

y_data = y_data.reshape(y_data.shape[0], 1)

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = tf.matmul(x, w) + b 
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 4e-1)
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)


train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(1001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

    if step % 10 == 0 :
        print(step, "cost :", cost_val)
        print("예측값")
        print(hy_val)
