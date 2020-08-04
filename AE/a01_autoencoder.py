#  copy 56, mnist auto encoder 적용

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# print('x_train : ', x_train[0])
# print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )

# print(x_train[0].shape)
# print(y_train[0])
# plt.imshow(x_train[0], 'gray') 
# plt.imshow(x_train[0]
# plt.show()

# 데이터 전처리 1. 원핫인코딩
# y data
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)


# 데이터 전처리 2. 정규화
# x data
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

print(x_train.shape)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(111, input_shape = (28*28, )))
model.add(Dropout(0.2))          
model.add(Dense(133, activation = 'relu'))
model.add(Dropout(0.2)) 
model.add(Dense(155, activation = 'relu'))
model.add(Dropout(0.2))     
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation = 'softmax'))

model.summary()
# 3. compile, 훈련
from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=200, validation_split = 0.2, callbacks = [early_stopping], verbose = 1) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ' , acc)

y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

# acc 98% 목표
'''
model.add(Dense(77, input_shape = (28*28, )))     
model.add(Dense(33, activation = 'relu'))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
epoch 100, batch_size 200
acc :  0.9722999930381775
'''
'''
위와 동일조건 epoch 80
acc :  0.9739000201225281
'''
'''
model = Sequential()
model.add(Dense(111, input_shape = (28*28, )))
model.add(Dropout(0.2))          
model.add(Dense(88, activation = 'relu'))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.2))     
model.add(Dense(55, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()
epoch = 95, batch_size = 200
acc :  0.9801999926567078 (!!!!)

2차 : acc :  0.98089998960495
'''
