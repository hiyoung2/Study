import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

# 레이어 구성 시 필요한 것 : 레이어마다 존재하는 weight, bias와 레이어마다 적용되는 활성화함수
# hypothesis도 마찬가지로 레이어마다 갱신되는데 이름을 살짝 바꿔서 아래와같이 준비해본다
# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)


# 첫 번째 레이어에 들어가는 weight, bias 준비
w1 = tf.Variable(tf.random_normal([2, 100]), name = 'weight1') # 100개의 노드를 준다
b1 = tf.Variable(tf.random_normal([100]), name = 'bias')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
# layer1 = tf.matmul(x, w1) + b1

# 위의 내용은
# model.add(Dense(100, activation = 'sigmoid', input_dim = 2)) 과 같음


w2 = tf.Variable(tf.random_normal([100, 50]), name = 'weight2') # 100개의 노드를 준다
b2 = tf.Variable(tf.random_normal([50]), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
# layer2 = tf.matmul(layer1, w2) + b2
# model.add(Dense(50, activation = 'sigmoid'))

# 맨 마지막에만 sigmoid 주고 나머지는 dafault 인 linear 적용하니까 Accuracy가 0.5에서 안 움직임
w3 = tf.Variable(tf.random_normal([50, 1]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)
# hypothesis = tf.matmul(layer2, w3) + b3
# model.add(Dense(1, acitvaion = 'sigmoid)) # output layer 가 됨


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0 :
            print(step, "cost :", cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict = {x:x_data, y:y_data})
    print("\n Hypothesis :", h, "\n Correct (y) :", c, "\n Accuracy :", a)



