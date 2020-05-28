# 54 copy, CNN 함수형으로 만들어라
# 그리고 54, 56, 58 성능을 비교하고 acc가 가장 잘 나오는 모델을 끝까지 높여봐라

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# print('x_train : ', x_train[0])
# print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
# print(x_train.ndim)
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
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)

# 2. 모델 구성 / naming이 필수는 아니지만 여러 종류의 레이어가 들어갈 땐 알아볼 수 있게 예쁘게 정리해주면 좋다
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, LSTM

input1 = Input(shape = (28, 28))
dense1 = LSTM(3, activation = 'relu')(input1)     
dense2 = Dense(28, activation = 'relu')(dense1)
dense3 = Dense(784, activation = 'relu')(dense2)   
dense4 = Dense(28, activation = 'relu')(dense3)   
output1 = Dense(10, activation = 'softmax')(dense4)

model = Model(inputs = input1, outputs = output1)


model.summary()

# 3. compile, 훈련
from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=200, validation_split = 0.3) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ' , acc)

# x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

# acc : 95% 맞추기

'''
input1 = Input(shape = (28, 28))
dense1 = LSTM(11, activation = 'relu')(input1)     
dense2 = Dense(55, activation = 'relu')(dense1)
dense3 = Dropout(0.2)(dense2)     

dense4 = Dense(33)(dense3)   
dense5 = Dense(77)(dense4)
dense6 = Dropout(0.2)(dense5)          

dense7 = Dense(44, activation = 'relu')(dense6)
dense8 = Dense(22)(dense7)
dense9 = Dropout(0.2)(dense8)

output1 = Dense(10, activation = 'softmax')(dense9)

model = Model(inputs = input1, outputs = output1)

model.summary()

epoch = 50, batch_size = 200

acc :  0.9531999826431274
'''