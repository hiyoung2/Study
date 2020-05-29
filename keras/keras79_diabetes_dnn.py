import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout


dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x)
print(y)

print('x.shape : ', x.shape)
print('y.shape : ', y.shape)

# 데이터 전처리
'''
scaler = MinMaxScaler()
scaler.fit(x)
scaler.fit(y)
x = scaler.transform(x)
y = scaler.transform(y)

print('x_scaled.shape : ', x.shape)
print('y_scaled.shape : ', y.shape)
'''


'''
# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

print(x_train.shape)  # (353, 10)
print(x_test.shape)   # (89, 10)
print(y_train.shape)  # (353, 347)
print(y_test.shape)   # (89, 347)


# 2. 모델 구성
model = Sequential()

model.add(Dense(33, input_shape = (10, )))
model.add(Dense(55))
model.add(Dense(99))
model.add(Dense(77))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(347, activation = 'sigmoid'))

model.summary()


# 3. 컴파일, 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 1, validation_split = 0.2, verbose= 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
print(y_pred)
print(np.argmax(y_pred, axis = 1))
'''