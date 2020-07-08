# mv? multi variable

import tensorflow as tf
tf.set_random_seed(777) # 잭 팟

# input : 3, output : 1
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# x가 3개이므로 각각의 weight가 존재
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# column이 여러 개인 것과 같으니까
# 각각의 가중치를 곱해서 마지막에 bias를 더해주면 된다

hypothesis = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
# hypothesis = tf.matmul([(x1, w1), (x2, w2), (x3, w3)]) + b
# hypothesis = tf.matmul([(x1, w1), (x2, w2), (x3, w3)]) + b


cost = tf.reduce_mean(tf.square(hypothesis - y))

# 이전까지는 train = tf.train 에다가 옵티, 미니마이즈 다 붙여썼는데
# 아래와 같은 식으로도 쓸 수 있다
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000044)

# 에러 발생 -> learning_rate 튜닝(대세)
# (learning_rate = 0.000009)
# 2000 cost : 5.4283814
# 2000 cost : 5.4283814
#   [147.98495 186.99979 179.57817 195.78354 144.60718]

# (learning_rate = 0.0000099)
# 2000 cost : 4.947587
#   [148.14153 186.8927  179.62646 195.81604 144.46869]

# (learning_rate = 0.00000999)
# 2000 cost : 4.902023
#   [148.15678 186.88226 179.63116 195.81918 144.45523]

# (learning_rate = 0.000009999)
# 2000 cost : 4.8975077
#   [148.1583  186.88124 179.63162 195.8195  144.45387]

# (learning_rate = 0.00001)
# 2000 cost : 4.897013
#   [148.15846 186.88112 179.63167 195.81955 144.45374]

# (learning_rate = 0.000019)
# 2000 cost : 2.0028448
#   [149.36345 186.05745 180.0039  196.06503 143.39297]

# (learning_rate = 0.00002)
# 2000 cost : 1.8226696
#   [149.46484 185.98822 180.0353  196.08516 143.30432]

# (learning_rate = 0.00003)
# 2000 cost : 0.7786242
#   [150.22919 185.46678 180.27269 196.23169 142.64142

# (learning_rate = 0.00004)
# 2000 cost : 0.42169204
#   [150.678   185.16151 180.41324 196.30931 142.26122]

# learning_rate = 0.00005)
# 여기서부터 커지면 nan 발생

# 0.00005 ~ 0.000047 : nan

# (learning_rate = 0.000046)
# 여기서부터 nan 등장 X

############################################# 최저 cost
# (learning_rate = 0.000045)
# 2000 cost : 0.3438245
#   [150.82773 185.06    180.46057 196.33192 142.1378

# (learning_rate = 0.000044)
# 2000 cost : 0.3563115
#   [150.80089 185.07823 180.45207 196.32811 142.15982]

# 에러 발생 -> optimizer 건드려 봄(선생님이 바라는 방향은 아닐 듯)
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.9) # 돌아감
# 2000 cost : 0.12410829
# [151.65028 184.5555  180.51636 196.10304 142.15302]

# optimizer = tf.train.AdadeltaOptimizer(learning_rate = 0.01) # 돌아감 # cost 상태 안 좋음

# optimizer = tf.train.AdagradOptimizer(learning_rate = 0.9) # 돌아감
# 2000 cost : 1.5924785
# [149.6915  185.8547  180.03098 195.98517 143.37895]


train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})

    if step % 10 == 0 :
        print(step, "cost :", cost_val, "\n ", hy_val)