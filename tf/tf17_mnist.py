import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# def min_max_scaler(dataset) :
#     numerator = dataset - np.min(dataset, 0) # axis = 0 : 열
#     denominator = np.max(dataset, 0) - np.min(dataset, 0)
#     return numerator / (denominator + 1e-7)

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
y_train=y_train.reshape(y_train.shape[0],10)
y_test=y_test.reshape(-1,10)


'''
# keras에서 OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
'''

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype('float32')/255

print("x_train.shape :", x_train.shape) # x_train.shape : (60000, 28, 28)    
print("x_test.shape :", x_test.shape)   # x_test.shape : (10000, 28, 28)
print("y_train.shape :", y_train.shape) # y_train.shape : (60000, 10)
print("y_test.shape :", y_test.shape)   # y_test.shape : (10000, 10)




w = tf.Variable(tf.zeros([28*28,10]),name="weight")
b = tf.Variable(tf.zeros([10]),name="bias")

# 파라미터들 변수로 미리 정의
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size) # 60000 / 100 = 600

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) # dropout (드롭아웃비율, placeholder사용해서 지정해줄 것)

# get_variable : Variable의 업그레이드형(성능이 더 좋다, 별 차이는 없다? 무슨 말,,,)
# get_variable : 초기변수가 없으면 자체 생성한다
# initializer, regularizer 등 사용 가능

# w1 = tf.Variable(tf.random_normal([28*28*, 512]), name = 'weight')
w1 = tf.get_variable("w1", shpae = [28*28, 512], 
                     initializer=tf.contrib.layer.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512])) 
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

w2 = tf.get_variable("w2", shpae = [512, 512], 
                     initializer=tf.contrib.layer.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512])) 
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

w3 = tf.get_variable("w3", shpae = [512, 512], 
                     initializer=tf.contrib.layer.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512])) 
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)


