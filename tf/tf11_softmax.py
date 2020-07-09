# 2020.07.09

import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1], # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0], # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0], # 0
          [1, 0, 0]]

# x_data와 y_data를 보고 x, y, w, b 각각 placeholder, Variable에 어떻게 넣어야 할 지 생각해보자

x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name = 'weight') # weight는 행렬곱으로 계산되기 때문에 무조건 4가 와야 함
# random_normal로 하지 않고 [4, 3] 4 x 3 형태의 데이터를 넣어도 된다
# 경사하강법으로 어차피 weight는 갱신되고 최적의 weight를 찾기 때문

# wx 계산 후 + bias
# w * x = [None, 3]
# 여기에 bias를 더해줌
# bias = [None, 3] 
# [1, 3], [3]이 잘 돌아감
# [3] 이 현재 이 소스에서는 cost 가 더 낮게 잘 나온다
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias') # bias 는 [1]? [3]? [1, 3]?

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-2).minimize(cost) # keras의 compile 과정이라 보면 된다

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0 :
            print(step, cost_val)

# 현재, 여기까지 최적의 weight와 bias가 구해져 있다

# hypothesis 훈련된 최적의 weight를 포함하고 있음
# 새로운 데이터 x를 넣었을 때 예측값
    a = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1))) # 1: 행을 의미

    b = sess.run(hypothesis, feed_dict={x:[[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1))) 

    c = sess.run(hypothesis, feed_dict={x:[[11, 33, 4, 13]]})
    print(c, sess.run(tf.argmax(c, 1)))

    # a, b, c를 넣어서 완성, 아래 소스를 수정해서 완성
    # all = sess.run(hypothesis, feed_dict = {x: [a, b, c]})

    # a, b, c 못 넣고 일단 돌림(실행은 되나, 옵션 충족 못함)
    # all = sess.run(hypothesis, feed_dict={ x : [[1, 11, 7, 9], [1, 3, 4, 3], [11, 33, 4, 13]]})
    
    # 2차 시도 그냥 아무렇게나 해 봄
    # all = sess.run(hypothesis, feed_dict = {x: a[0], b[1], c[2]})
    
    # print(all, sess.run(tf.argmax(all, 1)))