# 95번을 불러와서 모델을 완성하시오

# x_train = np.load('./data/mnist_train_x.npy')

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

dataset = np.load('./data/iris.npy')
# 현재 작업 공간 study의 data폴더 안의 iris.npy 라는 파일을 데이터세트로 쓴다

print(dataset)
# 현재 데이터세트를 확인

x = dataset[:, :4]
y = dataset[:, 4:]
# 지금 x data, ydata가 따로 분리가 안 되어 있기 때문에 슬라이싱으로 분리해줘야 한다
# 마지막 column이 y data

print(x)
print(y)
# x data, y data 확인

print('x.shape : ', x.shape) # (150, 4)
print('y.shape : ', y.shape) # (150, 1)
# data 처리하기 전에 shape 확인


# 데이터 전처리
# y data
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (150, 3)
# one-hot encoding으로 y data의 모양을 맞춰준다

# x data
scaler = MinMaxScaler()
scaler.fit(x)

print(x)
print(x.shape)

x = scaler.transform(x)

print(x)
print(x.shape)
# MinMax 스케일러로 x data 전처리 해 준다

print('x_caled.shape : ', x.shape)
# 스케일링을 거친 데이터 모양을 확인




# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 77, shuffle = True
)

print('x_train.shape : ', x_train.shape) # (120, 4)
print('x_test.shape : ', x_test.shape)   # (30, 4)
# 따로 train data, test data가 분리 안 되었기 때문에 
# sklearn의 model_selection 의 train_test_split을 통해 분리해준다


# 2. 모델 구성

model = Sequential()

model.add(Dense(100, input_shape = (4, )))
model.add(Dense(150))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 모델 구성 완료


# 3. 컴파일, 훈련

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 1)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
# print(y_pred)
print(np.argmax(y_pred, axis = 1))




'''
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(99))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

epo 100, batch = 1
loss :  0.1337626874446869
acc :  0.9333333373069763
'''