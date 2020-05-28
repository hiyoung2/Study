# 기본 CNN MODEL

import numpy as np
from keras.datasets import cifar10 
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Conv2D, Dense, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

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
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(x_train.shape)

# 2. 모델 구성

input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(50, (2, 2))(input1)
dense2 = Dropout(0.3)(dense1)
dense3 = Conv2D(70, (3, 3), activation = 'relu')(dense2)
dense4 = Dropout(0.2)(dense3)

dense5 = Conv2D(90, (3, 3), padding = 'same', activation = 'relu')(dense4)
dense6 = Dropout(0.2)(dense5)
dense7 = MaxPooling2D(pool_size = 2)(dense6)


dense8 = Conv2D(80, (2, 2), padding = 'same', activation = 'relu')(dense7)
dense9 = Dropout(0.2)(dense8)
dense10 = MaxPooling2D(pool_size = 2)(dense9)


dense11 = Flatten()(dense10)
output1 = Dense(10, activation = 'softmax')(dense11)

model = Model(inputs = input1, outputs = output1)

model.summary()

# 3. 컴파일, 훈련
# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'auto') 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=95, batch_size=100, validation_split = 0.3, verbose = 1) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)

print('loss : ', loss)
print('acc : ' , acc)

y_pred = model.predict(x_test)

print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

'''
1차 
acc :  0.785099983215332 엉망
2차
acc :  0.7825999855995178
'''