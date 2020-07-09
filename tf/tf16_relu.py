# mnist : 다중분류

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# def min_max_scaler(dataset) :
#     numerator = dataset - np.min(dataset, 0) # axis = 0 : 열
#     denominator = np.max(dataset, 0) - np.min(dataset, 0)
#     return numerator / (denominator + 1e-7)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape :", x_train.shape) # x_train.shape : (60000, 28, 28)    
print("x_test.shape :", x_test.shape)   # x_test.shape : (10000, 28, 28)
print("y_train.shape :", y_train.shape) # y_train.shape : (60000,)
print("y_test.shape :", y_test.shape)   # y_test.shape : (10000,)

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[1])


# MinMaxScaler 쓸 때
# x_train = min_max_scaler(x_train)
# x_test = min_max_scaler(x_test)

x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])

# w = tf.Variable(tf.random_normal([28*28, 10]), name = 'weight')
# b = tf.Variable(tf.random_normal([10]), name = 'bias')

# 2. 모델 구성

# hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

w1 = tf.Variable(tf.zeros([28*28, 10]), name = 'w1')
b1 = tf.Variable(tf.zeros([10]), name = 'b1')
hypothesis = tf.nn.softmax(tf.matmul(x, w1) + b1)

# w2 = tf.Variable(tf.zeros([128, 64]), name = 'w2')
# b2 = tf.Variable(tf.zeros([64]), name = 'b2')
# layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
# # layer2 = tf.nn.elu(tf.matmul(x, w1) + b1)
# # layer2 = tf.nn.selu(tf.matmul(x, w1) + b1)

# w3 = tf.Variable(tf.zeros([64, 32]), name = 'w3')
# b3 = tf.Variable(tf.zeros([32]), name = 'b3')
# layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

# layer4 = tf.nn.dropout(layer3, keep_prob = 0.2)

# w4 = tf.Variable(tf.zeros([32, 10]), name = 'w4')
# b4 = tf.Variable(tf.zeros([10]), name = 'b4')
# hypothesis = tf.nn.softmax(tf.matmul(layer4, w4) + b4)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(101) :
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_train, y:y_train}) #fit
        if step % 10 == 0 :
            print(step, cost_val)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

