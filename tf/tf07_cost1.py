# 그림을 그려보기 위해 만든 파일

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]

w = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y))

# model.fit 에 반환되는 것들 : history
# 아래 : history를 구현하겠다는 뜻

w_history = []
cost_history = []

with tf.Session() as sess :
    for i in range(-30, 50) :
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict ={w : curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

# w_history : 가중치가 변화되는 모양새를 보여줌
plt.plot(w_history, cost_history)
plt.show()

# 그림 결과 : 이차함수
# 세로축 : cost, 가로축 : weight

# keras에서는 history에서 제공해주는 기능을
# tensorflow 1ver에서는 풀어서 쓴 것