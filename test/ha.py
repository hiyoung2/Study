'''
print(hite)
print(samsung)
print('hite.shape : ', hite.shape)
print('samsung.shape : ', samsung.shape)

print(hite.head())
print(samsung.tail())

print(hite.values)
print(samsung.values)

aaa = hite.values
print(aaa)
bbb = samsung.values
print(bbb)

# 1. 1 데이터 npy로 저장
np.save('./data/hite.npy', arr = aaa)
np.save('./data/samsung.npy', arr = bbb)


import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# 1.2 데이터 준비
x = np.load('./data/hite.npy', allow_pickle=True)
y = np.load('./data/samsung.npy', allow_pickle=True)


print(x)
print(y)

print("=============================")
x = x[1:509]
y = y[1:509]

print(x)
print(y)

# print('x.shpae : ', x.shape)
# print('y.shape : ', y.shape)



# 1.3 데이터 전처리
# MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)

# print(x)


# 1.4 x1, x2, y1


# 1.4 train, test split










# 2. 모델 구성

# 3. 컴파일, 훈련

# 4. 평가, 예측
'''