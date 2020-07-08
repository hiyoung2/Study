# x_train 과  y_train에  placeholer 를 적용 시켜 소스 완성
# feed_dict를  with 문 안에 있는 sees.run 부분에 넣으면 된다!
# tf04_placeholder 파일에 보면 
# >>> sess.run 할 때에 feed_dict를 쓰면 된다
# 라고 했으므로!

import tensorflow as tf
tf.set_random_seed(777)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) : 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 

        if step % 20 == 0 : 
            print(step, cost_val, W_val, b_val) 

