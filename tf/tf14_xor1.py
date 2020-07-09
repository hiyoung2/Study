# 인공지능의 겨울
# 어떻게 해결? 딥러닝에선 레이어를 추가해주었다
# tensorflow에서 어떻게 레이어를 추가할까? -> 문법적으로 배우면 된다!
# xor2 파일로 가시오!!

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

# 소스 완성
# x, y, w, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy 준비해놓을 것

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3)

train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# 0.5 초과하면 1, 0.5 이하면 0 -> sigmoid
# tf.cast() : true/ false를 1.0 / 0.0 으로 반환
# hypothesis > 0.5 == true == 1.0 반환
# hypothesis =< 0.5 == false == 0.0 반환

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0 :
            print(step, "cost :", cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict = {x:x_data, y:y_data})
    print("\n Hypothesis :", h, "\n Correct (y) :", c, "\n Accuracy :", a)



