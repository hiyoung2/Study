# 54 copy, 레이어 그대로 두고 함수형으로 만들어라

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print('x_train : ', x_train[0])
print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )

print(x_train[0].shape)
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
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape)

# 2. 모델 구성 / naming이 필수는 아니지만 여러 종류의 레이어가 들어갈 땐 알아볼 수 있게 예쁘게 정리해주면 좋다
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Dropout, MaxPooling2D, Flatten

input1 = Input(shape = (28, 28, 1))
dense1 = Conv2D(77, (2, 2), name = 'Conv2D_1')(input1)     
dense2 = Conv2D(111, (3, 3), activation = 'relu', name = 'Conv2D_2')(dense1)
dense3 = Dropout(0.2, name = 'Dropout_1')(dense2)     

dense4 = Conv2D(99, (3, 3) , padding = 'same', name = 'Conv2D_3')(dense3)   
dense5 = MaxPooling2D(pool_size = 2, name = 'MaxPooling2D_1')(dense4)
dense6 = Dropout(0.2, name = 'Dropout_2')(dense5)          

dense7 = Conv2D(55, (2, 2), padding = 'same', activation = 'relu', name = 'Conv2D_4')(dense6)
dense8 = MaxPooling2D(pool_size = 2, name = 'MaxPooling2D_2')(dense7)
dense9 = Dropout(0.2, name = 'Dropout_3')(dense8)

dense10 = Flatten(name = 'Flatten')(dense9)
output1 = Dense(10, activation = 'softmax')(dense10)

model = Model(inputs = input1, outputs = output1)

model.summary()

# 3. compile, 훈련
# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=5, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=70, batch_size=200, validation_split = 0.2) 

# 4. 예측, 평가

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)

print('loss : ', loss)
print('acc : ' , acc)

# x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

