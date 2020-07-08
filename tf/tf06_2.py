# tf06_01.py를 copy
# lr을 수정해서 연습
# 0.01 -> 0.1 / 0.001 / 1
# epoch 2000번보다 줄여서

import tensorflow as tf
tf.set_random_seed(777)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

# placeholder로 선언
x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.07).minimize(cost)

with tf.Session() as sess : # 
    sess.run(tf.global_variables_initializer()) 

    for step in range(101) : 
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 

        if step % 20 == 0 : 
            print(step, cost_val, W_val, b_val) 

    # predict !
    print("예측(4) :", sess.run(hypothesis, feed_dict = {x_train :[4]})) 
    print("예측(5, 6) :", sess.run(hypothesis, feed_dict = {x_train : [5, 6]}))
    print("예측(6, 7, 8) :", sess.run(hypothesis, feed_dict = {x_train : [6, 7, 8]}))

# 원래 처음 소스 epo = 2000, lr = 0.01
# 예측(4) : [9.000078]
# 예측(5, 6) : [11.000123 13.000169]
# 예측(6, 7, 8) : [13.000169 15.000214 17.000257]

# epoch 줄이고, lr 다양하게 시도

# epo = 1000

# lr = 0.1    
# 예측(4) : [9.000002]
# 예측(5, 6) : [11.000003 13.000004]
# 예측(6, 7, 8) : [13.000004 15.000005 17.000006]

# lr = 0.001
# 예측(4) : [9.007341]
# 예측(5, 6) : [11.01162  13.015898]
# 예측(6, 7, 8) : [13.015898 15.020176 17.024454]

# lr = 1.0
# nan 값?

# epo = 500
# lr = 0.09
# 예측(4) : [9.000002]
# 예측(5, 6) : [11.000003 13.000005]
# 예측(6, 7, 8) : [13.000005 15.000006 17.000008]

# lr = 0.0001 -> 과소적합
# 예측(4) : [5.946941]
# 예측(5, 6) : [7.2625985 8.578257 ]
# 예측(6, 7, 8) : [ 8.578257  9.893914 11.209572] 

# epo = 300
# lr = 0.07
# 예측(4) : [9.000057]
# 예측(5, 6) : [11.000091 13.000123]
# 예측(6, 7, 8) : [13.000123 15.000156 17.00019 ]

# epo = 100
# lr = 0.07
# 예측(4) : [9.001699]
# 예측(5, 6) : [11.002684 13.003668]
# 예측(6, 7, 8) : [13.003668 15.004652 17.005636]