# 넌 정말 sigmoid
# 난 정말 sigmoid

import tensorflow as tf
tf.set_random_seed(777) 

x_data = [[1, 2],
          [2, 3], 
          [3, 1],
          [4, 3],  
          [5, 3],
          [6, 2]]

y_data = [[0], 
          [0], 
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])          
y = tf.placeholder(tf.float32, shape=[None, 1])          

# w는 x와 곱해지는 연산의 대상, 따라서 행렬의 곱셈 법칙에 따라  x 데이터의 열과 같은 수의 행으로 설정해주어야 한다
w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# tf.matmul : 행렬의 곱셈을 해 주는 함수
# x = 5 by 3, w = 3 by 1 
# x * w = 5 by 1
# wx + b = 5 by 1 + 1 = 5 by 1
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b

# activation : 결괏값을 활성화함수에 통과시켜서 다음 레이어에 넘겨줌
# tensorflow에서는 활성화함수를 hypothesis에 wrap 해 주면 된다

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))
# sigmoid : 분류 쪽에서 사용하는 활성화 함수
# sigmoid 사용하는 cost 정의?
# sigmoid 를 사용했을 때의 cost의 식은 이러하다, 정도까지만 이해


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000045)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

    if step % 10 == 0 :
        print(step, "cost :", cost_val)
        print("예측값")
        print(hy_val)