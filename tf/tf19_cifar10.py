import tensorflow as tf
import numpy as np

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

# 1. 데이터
# keras의 datasets를 불러오는 방법
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train.shape :", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape :", x_test.shape)   # (10000, 32, 32, 3)
print("y_train.shape :", y_train.shape) # (50000, 1))
print("y_test.shape :", y_test.shape)   # (10000, 1)

# 1-1. 데이터 전처리 : One-Hot Encoding
# cifar10의 최종 output node : 10(10가지 종류)
# airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks

# (1) tensorflow 에서 제공하는 one-hot 사용

# tf을 기반으로 한 one_hot이기 때문에 Session을 통과시켜줘야 한다
# sess.close()사용하지 않는 대신 with문을 사용
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train, 10)) # depth = 10 이렇게 대신 그냥 10만 적어도 된다
    y_test = sess.run(tf.one_hot(y_test, 10))
y_train = y_train.reshape(-1, 10) # one_hot 사용 후, reshape을 해 줘야 한다
y_test = y_test.reshape(-1, 10)

# (2) keras에서 제공하는 one-hot 사용
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 1-2. 데이터 전처리 : 정규화
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 파라미터들 변수로 미리 정의

# learning_rate, 학습률
lr = 0.01
# epoch
training_epoch = 100
# batch_size
batch_size = 100
total_batch = int(len(x_train)/batch_size)
# dropout
# tf에서는 정한 비율만큼 drop 시키는 것이 아니라 keep 한다
# 즉, keep_prob에 feed_dict로 0.7을 입력하면 70%를 사용, 30%를 drop한다
keep_prob = tf.placeholder(tf.float32)

# 2. 모델 구성
# 2-1. x, y를 placeholder로 준비
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
# x = tf.placeholder(tf.float32, shape = [None, 32*32*3])
x = tf.reshape(x, [-1, 32, 32, 3])

y = tf.placeholder(tf.float32, shape = [None, 10])


# 2-2. w, b를 variable로 만들면서 레이어 구성
w1 = tf.get_variable("w1", shape = [3, 3, 3, 32]) # kernel_size(filter) : (3, 3) / channel(color) : 3 / outputnode : 32
L1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = (1, 2, 2, 1), strides = [1, 2, 2, 1], padding = 'SAME')

w2 = tf.get_variable("w2", shape = [3, 3, 32, 64])
L2 = tf.nn.conv2d(L1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = (1, 2, 2, 1), strides = [1, 2, 2, 1], padding = 'SAME')

# Flatten()
print("L2 :", L2) # shape=(?, 8, 8, 32)
L2_flat = tf.reshape(L2, [-1, 8*8*64])

# Flatten() 이후 Dense 레이어 구성
w3 = tf.get_variable("w3", shape = [8*8*64, 128], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]))
# b3 = tf.Variable(tf.zeros([10]))
# b3 = tf.Variable(tf.random_uniform([10]))

L3 = tf.matmul(L2_flat, w3) + b3

w4 = tf.get_variable("w4", shape = [128, 64], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([64]))
L4 = tf.matmul(L3, w4) + b4

w5 = tf.get_variable("w5", shape = [64, 32], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([32]))
L5 = tf.matmul(L4, w5) + b5

w6 = tf.get_variable("w6", shape = [32, 10], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L5, w6) + b6)
# 최종 output laeyer : hypothesis



# 3. 컴파일 및 훈련
# cost(loss) , 손실 함수 측정 : categorical_crossentropy(다중 분류에서 사용하는)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch) :
        avg_cost = 0

        for i in range(total_batch) :
            start = i * batch_size
            end = start + batch_size

            batch_xs, batch_ys  = x_train[start:end], y_train[start:end]

            feed_dict = {x:batch_xs, y:batch_ys, keep_prob : 0.7}

            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)

            avg_cost = c / total_batch
        
        print('EPOCH :', '%04d' % (epoch +1), 'COST =', '%.9f' % (avg_cost))


    print("훈련 완료")


    # 4. 평가 및 예측
    # prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
    prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    print("ACC : ", sess.run(accuracy, feed_dict = {x:x_test, y:y_test, keep_prob : 1.0}))




'''
동일 레이어, 노드 구성에 random_normal vs zeros
b3 = tf.Variable(tf.random_normal([10]))
ACC :  0.2433

b3 = tf.Variable(tf.zeros([10]))
ACC :  0.2379
'''


'''
lr = 0.01
epo = 100
ACC :  0.6782
'''

'''
현재 소스 상태에서
epo 67부터 COST nan으로 출력됨
'''