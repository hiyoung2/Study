import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# 1. DATASET 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("x_train.shape :", x_train.shape) # x_train.shape : (60000, 28, 28)    
# print("x_test.shape :", x_test.shape)   # x_test.shape : (10000, 28, 28)
# print("y_train.shape :", y_train.shape) # y_train.shape : (60000,)
# print("y_test.shape :", y_test.shape)   # y_test.shape : (10000,)

# 데이터 전처리 1.  OneHotEncoding
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train = y_train.reshape(y_train.shape[0],10)
y_test = y_test.reshape(-1,10)


'''
# keras에서 OneHotEncoding 해도 됨
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
'''

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype('float32')/255
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1).astype('float32')/255

print("x_train.shape :", x_train.shape) # x_train.shape : (60000, 28, 28, 1)    
print("x_test.shape :", x_test.shape)   # x_test.shape : (10000, 28, 28, 1)
print("y_train.shape :", y_train.shape) # y_train.shape : (60000, 10)
print("y_test.shape :", y_test.shape)   # y_test.shape : (10000, 10)



# 파라미터들 변수로 미리 정의
lr = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size) # 60000 / 100 = 600

# DNN Model input
x = tf.placeholder(tf.float32, shape=[None,28*28])

# CNN Model 구성을 위한 단계
# tensorflow에도 reshape 있음
x_img = tf.reshape(x, [-1, 28, 28, 1]) # -1 : 전체행(60000)
# -> input shape

# y는 따로 shape 처리 해 줄 것 없음
y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) # dropout (드롭아웃비율설정해주는 것과 같은데 반대개념!, placeholder사용해서 지정해줄 것)

# w1 = tf.Variable(tf.random_normal([28*28*, 512]), name = 'weight') 위 와래 같은 것
w1 = tf.get_variable("w1", shape = [3, 3, 1, 32]) # 3, 3 : keras로 따지면 kernel_size / 1 : color / 32 : outputnode
print('------------------------------------------')
print('w1 :', w1) # # w1 : <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>
# Conv2D(output node(32), (kernel ex. (3, 3), input_shape = (28, 28, 1)))
# input_shape는 위에서 설정된 것

# b1는 어디로?
# Conv2D 레이어 자체에 b1는 알아서 연산이 된다
# Conv2D 레이어 구성시에 앞선 DNN 모델처럼 bias는 따로 설정해주지 않아도 된다

L1 = tf.nn.conv2d(x_img, w1, strides =[1, 1, 1, 1], padding = 'SAME') 
# strides 가장 왼쪽, 오른쪽 1 : 형식적으로 써 주는 것 / mnist 실질적으로 사용되는 부분은 중간 1, 1
print('------------------------------------------')
print("L1 :", L1) # L1 : Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

# padding = 'VALID' 이면
# L1 : Tensor("Conv2D:0", shape=(?, 26, 26, 32), dtype=float32)
L1 = tf.nn.selu(L1) # L1을 selu 활성화함수에 통과시킨다 # x와 w는 이미 위의 L1에서 연산이 됨

L1 = tf.nn.max_pool(L1, ksize = (1, 2, 2, 1), strides = [1, 2, 2, 1], padding = 'SAME') # Maxpooling에도 strides가 적용된다(keras에서 써 보진 않음)
print('------------------------------------------')
print("L1(maxpool) :", L1) # L1(maxpool) : Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L1 = tf.nn.dropout(L1, keep_prob = keep_prob)



w2 = tf.get_variable("w2", shape = [3, 3, 32, 64]) # L1의 output node 32가 color 자리에 들어옴 # 64 : output node
# color 설정 이름은 channel이라고도 함
# shape = [kernel_size, channel, outputnode]
L2 = tf.nn.conv2d(L1, w2, strides =[1, 1, 1, 1], padding = 'SAME') 
L2 = tf.nn.selu(L2) 
L2 = tf.nn.max_pool(L2, ksize = (1, 2, 2, 1), strides = [1, 2, 2, 1], padding = 'SAME')

print("L2 :", L2) # L2 : Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# 평평하게 쫙 펴 줘야 함
L2_flat = tf.reshape(L2, [-1, 7*7*64])
print('------------------------------------------')
print("L2_flat :", L2_flat) # L2_flat : Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)


w3 = tf.get_variable("w3", shape = [7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10])) 
hypothesis = tf.nn.softmax(tf.matmul(L2_flat, w3) + b3)



'''
w2 = tf.get_variable("w2", shape = [512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512])) 
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)


w3 = tf.get_variable("w3", shape = [512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512])) 
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)


w4 = tf.get_variable("w4", shape = [512, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256])) 
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

w5 = tf.get_variable("w5", shape = [256, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10])) 
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)
'''

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for epoch in range(training_epochs) : # 15로 설정
    avg_cost = 0 # 평균 비용(= 평균 cost, 평균 loss)

    for i in range(total_batch) : # 600으로 설정
#############################이 부분 구현할 것##############################        
        # batch_xs, batch_y = x_train[0:100], y_train[0:100]
        # batch_xs, batch_y = x_train[100:200], y_train[100:200]

        # batch_xs, batch_ys = x_train[i:batch_size] # y_train은 따로 해 줄 필요가 없다
        # batch_xs, batch_ys = x_train[i+batch_size:batch_size +batch_size] # y_train은 따로 해 줄 필요가 없다
        # 이렇게 구현해야함

        # 어떻게 구성해야하는 지 과정 설명
        start = i * batch_size   #  0  100 200 ,,,(for문 실행시)
        end = start + batch_size # 100 200 300 ,,,(for문 실행시)

        batch_xs, batch_ys = x_train[start:end], y_train[start:end]

        # start = i * batch_size # 100
        # end = start + batch_size # 200

        # start = i * batch_size # 200
        # end = start + batch_size # 300

###########################################################################        
        feed_dict = {x_img:batch_xs, y:batch_ys, keep_prob:0.7} # keep_prob : 0.7은 70%를 남기겠다는 뜻(drop out 0.3을 주는 것과 같음)
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch

    print('Epoch :', '%04d' %(epoch + 1), 
          'cost = ', '{:.9f}'.format(avg_cost))

print("훈련 끝!")


prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print('ACC :', sess.run(accuracy, feed_dict = {x_img:x_test, y:y_test, keep_prob : 1}))

