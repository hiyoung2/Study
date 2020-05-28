# 54 copy, CNN 함수형으로 만들어라
# 58에서 함수형으로 만들어서 이번에 시퀀셜로 만들어 봄



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
# x_train = x_train.reshape(60000, 28, 28).astype('float32') / 255.
# x_test = x_test.reshape(10000, 28, 28).astype('float32') / 255.

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# 28, 28 로 input 넣을 거면 x data는 따로 reshape 해 줄 필요 없다! 
# 위에 보면 x data shape 자체가 3차원 이기 때문, lstm도 3차원, 차원이 같으므로 와꾸 안 맞춰도 됨
# 대신, 28, 28 이 아니라 784, 1 로 input을 넣을 때는 reshape 해 줘야 한다

print(x_train.shape)

# 2. 모델 구성 / naming이 필수는 아니지만 여러 종류의 레이어가 들어갈 땐 알아볼 수 있게 예쁘게 정리해주면 좋다
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, LSTM

model = Sequential()
model.add(LSTM(11, input_length=28 , input_dim =28, activation = 'relu'))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3. compile, 훈련
from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=200, validation_split = 0.3) 

# 4. 예측, 평가

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ' , acc)

# x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

'''
model = Sequential()
model.add(LSTM(11, input_length=28 , input_dim =28, activation = 'relu'))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
epoch = 50, batch_size = 200
acc :  0.9631999731063843
'''