import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. 데이터 준비

dataset = load_iris()
x = dataset.data
y = dataset.target

# print(x)
# print(y)

# 1.1 데이터 전처리
y = np_utils.to_categorical(y)

# 1.2 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print('x_transform.shape : ', x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size  = 0.8
)

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)


# 2, 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape = (4, 1), activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 30, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

y_pred = model.predict(x_test)
print(np.argmax(y_pred, axis = 1))
