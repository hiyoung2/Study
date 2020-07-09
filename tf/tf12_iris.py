# 다중분류
# train_test_split도 해야 함

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import tensorflow as tf

iris = load_iris()

print(type(iris))


x_data = iris['data']
y_data = iris['target']

print("x_data.shape :", x_data.shape) # x.shape : (150, 4)
print("y_data.shape :", y_data.shape) # y.shape : (150,)

# iris data는 one-hot encoding이 필요
# aaa = tf.one_hot(y, ???)

sess = tf.Session()
y_data = tf.one_hot(y_data, depth=3).eval(session=sess)
sess.close()

# y_train = tf.one_hot(y_train, depth = 3)
# y_train = tf.reshape(y_train, [120, 3])

# y_test = tf.one_hot(y_test, depth = 3)
# y_test = tf.reshape(y_test, [30, 3])

# print("x_train.shape :", x_train.shape)
# print("x_test.shape :", x_test.shape)
# print("y_train.shape :", y_train.shape)
# print("y_test.shape :", y_test.shape)

'''
x_train.shape : (120, 4)
x_test.shape : (30, 4)
y_train.shape : (120, 3)
y_test.shape : (30, 3)
'''

x_train = x_data[:120,:]
x_test = x_data[120:, :]

y_train = y_data[:120, ]
y_test = y_data[120:, ]


x = tf.placeholder(tf.float32, shape = [None, 4])
y = tf.placeholder(tf.float32, shape = [None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')

# x_train, x_test, y_train, y_test = train_test_split(
#     x_data, y_data, train_size = 0.8, random_state = 77, shuffle = True
# )

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.04).minimize(cost) # compile


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
    #     _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_train, y:y_train}) #fit

    #     if step % 200 == 0 :
    #         print(step, cost_val)
        sess.run(optimizer, feed_dict={x: x_train, y: y_train})

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
            
