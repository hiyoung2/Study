import tensorflow as tf
tf.set_random_seed(777) 

x_data = [[73, 51, 65],
          [92, 98, 11], 
          [89, 31, 33],
          [99, 33, 100],  
          [17, 66, 79]]

y_data = [[152], 
          [185], 
          [180],
          [205],
          [142]]

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000045)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000049)
# optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.1)



train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

    if step % 10 == 0 :
        print(step, "cost :", cost_val)
        print("예측값")
        print(hy_val)