# 2020.07.13

import tensorflow as tf
import numpy as np


# 1. 데이터
# data : hihello
# 인덱스를 넣어주기 위해 한 글자씩 뺐다(알파벳순)
idex2char = ['e', 'h', 'i', 'l', 'o']

# _data = np.array([['h'], ['i'], ['h'], ['e'], ['l'], ['l'], ['o']])
_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1, 1)

print("_data.shape :", _data.shape)
print(_data)
print(type(_data))
# _data.shape : (7, 1)
# [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
# <class 'numpy.ndarray'>

print("--------------------------------------------------")

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
print("_data.shape :", _data.shape)
print(_data)
print(type(_data))
print(_data.dtype)
print()

# _data.shape : (7, 5)
# [[0. 1. 0. 0. 0.]  h
#  [0. 0. 1. 0. 0.]  i
#  [0. 1. 0. 0. 0.]  h
#  [1. 0. 0. 0. 0.]  e
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 0. 1.]] o
# <class 'numpy.ndarray'>
# float64

# hihell 까지 x,  ihello 까지  y로 잡고 훈련을 시킨다

x_data = _data[:6, ] # hihell
y_data = _data[1:, ] #  ihello

print("x_data")
print(x_data)
print()
print("y_data")
print(y_data)

# x_data (6, 5) -> (1, 6, 5) (for LSTM) -> input shape = 6, 5 /(6, 5, 1) 이렇게 해도 LSTM 모델에 들어간다(나중에 시도해봐라)
# [[0. 1. 0. 0. 0.]  h
#  [0. 0. 1. 0. 0.]  i
#  [0. 1. 0. 0. 0.]  h
#  [1. 0. 0. 0. 0.]  e
#  [0. 0. 0. 1. 0.]  l
#  [0. 0. 0. 1. 0.]] l

# y_data (6, 5) y_data는 (1, 6)으로 준비해야 한다 -> index로 -> [[2, 1, 0, 3, 3, 4]]
# [[0. 0. 1. 0. 0.]  i  2(index자리)
#  [0. 1. 0. 0. 0.]  h  1
#  [1. 0. 0. 0. 0.]  e  0
#  [0. 0. 0. 1. 0.]  l  3
#  [0. 0. 0. 1. 0.]  l  3
#  [0. 0. 0. 0. 1.]] o  4

y_data = np.argmax(y_data, axis = 1)
print()
print("y_data(argmax)")
print(y_data)
print("y_data.shape :", y_data.shape)
print()
# [2 1 0 3 3 4]
# y_data.shape : (6,)

# RESHAPE
x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print("x_data.shape :", x_data.shape) # (1, 6, 5)
print("y_data.shape :", y_data.shape) # (1, 6)
print()

# shape 등을 변수로 지정해두고 사용
sequence_length = 6
input_dim = 5
output = 5
batch_size = 1 # 전체 행
lr = 0.1

# x = tf.placeholder(tf.float32, shape = [None, 6, 5])
# X = tf.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.placeholder(tf.int32, (None, sequence_length))

# for warning
X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X) # Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(Y) # Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)

# 2. 모델 구성

# keras 문법
# model.add(LSTM(output, input_shape = (6, 5)))

# tensorflow 문법
# cell = tf.nn.rnn_cell.BasicLSTMCell(output) # == model.add(LSTM(output, # LSTM의 outputnode 를 명시
cell = tf.keras.layers.LSTMCell(output)

hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

print("hypothesis :", hypothesis) # shape=(?, 6, 100) -> shape=(?, 6, 5) (output 변수 변경 후)


# 3-1. 컴파일
weights = tf.ones([batch_size, sequence_length]) # y의 shape와 같다
# LSTM 모델의 loss
# 자세한 수식 설명은 x, hypothesis - y 결과를 보여준다고 보면 된다(mse 처럼)
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights
)

cost = tf.reduce_mean(sequence_loss)

# train = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
# for warning
train = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost)

# axis = 0 : 1 / axis = 1 : 6 / axis = 2 : 5
prediction = tf.argmax(hypothesis, axis = 2)



# 3-2. 훈련
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for i in range(401) :
        loss, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})

        result = sess.run(prediction, feed_dict = {X:x_data})

        print(i, "loss :", loss, "prediction :", result, "true_Y :", y_data)

        result_str = [idex2char[c] for c in np.squeeze(result)]
        print("\nPredicton str :", ''.join(result_str))
