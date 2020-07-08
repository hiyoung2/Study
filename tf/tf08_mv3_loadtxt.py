# csv 데이터 불러오기

import numpy as np
import tensorflow as tf
tf.set_random_seed(777) 

# csv 파일이 수치로만 되어있으므로 판다스 없이 바로 numpy로 불러오면 된다
dataset = np.loadtxt('./data/csv/data-01-test-score.csv', delimiter = ',', dtype = np.float32)

x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]


x = tf.placeholder(tf.float32, shape=[None, 3])          
y = tf.placeholder(tf.float32, shape=[None, 1])          

# w는 x와 곱해지는 연산의 대상, 따라서 행렬의 곱셈 법칙에 따라  x 데이터의 열과 같은 수의 행으로 설정해주어야 한다
w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# tf.matmul : 행렬의 곱셈을 해 주는 함수
# x = 5 by 3, w = 3 by 1 
# x * w = 5 by 1
# wx + b = 5 by 1 + 1 = 5 by 1
hypothesis = tf.matmul(x, w) + b # wx+b

cost = tf.reduce_mean(tf.square(hypothesis - y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 5e-5)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000049)
# optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.1)

# (learning_rate = 0.00004)
# 2000 cost : 6.5554423

# (learning_rate = 0.00005)
# 2000 cost : 6.3071394

# (learning_rate = 0.00006)
# nan

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

    if step % 10 == 0 :
        print(step, "cost :", cost_val)
        print("예측값")
        print(hy_val)



# 이제껏 activation 주지 않았음 -> 디폴트로 linear 
# y = wx + b -> 선형 모델 따라서 linear가 적용된 것