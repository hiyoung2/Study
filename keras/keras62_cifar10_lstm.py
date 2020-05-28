# 기본 CNN MODEL

from keras.datasets import cifar10 
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Conv2D, Dense, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape : ", x_test.shape)   # (10000, 32, 32, 3)
print("y_train.shape : ", y_train.shape) # (50000, 1)
print("y_test.shape : ", y_test.shape)   # (10000, 1)

# plt.imshow(x_train[3])
# plt.show()

# 데이터 전처리
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 정규화
# x_train = x_train.reshape(50000, 32*3, 32).astype('float32') / 255.0
# x_test = x_test.reshape(10000, 32*3, 32).astype('float32') / 255.0

x_train = x_train.reshape(50000, 64, 48).astype('float32') / 255.0
# x_test = x_test.reshape(10000, 64, 48).astype('float32') / 255.0

print(x_train.shape)

# 2. 모델 구성
input1 = Input(shape = (64, 48))
dense1 = LSTM(10, activation = 'relu')(input1)     
dense2 = Dense(18, activation = 'relu')(dense1)
dense3 = Dropout(0.2)(dense2)
dense4 = Dense(20, activation = 'relu')(dense3)  
dense5 = Dropout(0.2)(dense4) 
dense6 = Dense(14, activation = 'relu')(dense5)
dense7 = Dropout(0.2)(dense6)   
output1 = Dense(10, activation = 'softmax')(dense7)

model = Model(inputs = input1, outputs = output1)

model.summary()

# 3. 컴파일, 훈련
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto') 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=200, validation_split = 0.3, verbose = 1) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ' , acc)

y_pred = model.predict(x_test)

# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)
