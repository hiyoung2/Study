import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. 데이터 준비
diabetes = load_diabetes()
x = diabetes['data']
y = diabetes['target']

print(x)
print(y)

print('x.shape : ', x.shape) # (442, 10)
print('y.shape : ', y.shape) # (442,)

# 1.1 데이터 전처리 - Scaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# 1.2 데이터 분리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y , train_size = 0.8
)

print('x_train.shape : ', x_train.shape) # (353, 10)
print('x_test.shape : ', x_test.shape)   # (89, 10)

# 1.3 데이터 shape 맞추기
x_train = x_train.reshape(x_train.shape[0], 5, 2)
x_test = x_test.reshape(x_test.shape[0], 5, 2)


# 2. 모델 구성
model = Sequential()

model.add(LSTM(10, input_shape = (5, 2)))
# model.add(Dropout(0.2))
model.add(Dense(15))
# model.add(Dropout(0.2))
model.add(Dense(19))
# model.add(Dropout(0.2))
model.add(Dense(21))
# model.add(Dropout(0.3))
model.add(Dense(17))
# model.add(Dropout(0.3))
model.add(Dense(11))
# model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dense(5))
# model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 200, batch_size = 1, validation_split = 0.3, verbose = 1)

# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)
# print(y_pred)

# RMSE, R2

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('RSME : ', RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print('R2 : ', r2)
