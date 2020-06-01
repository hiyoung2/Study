# mnist 파일을 npy로 저장하기

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

# 데이터 전처리 하기 전에 데이터들을 저장
# 데이터 전처리 하고 난 후에 저장해도 상관은 없지만
# 나중에 전처리 부분에 문제가 생겼다면 좀 난감하다
# 데이터 전처리 하기 전에 데이터들을 저장해주면 모델 짤 때 전처리를 해 줘야 한다

# numpy 자체가 행렬, 배열? / numpy 변수명 arr 자잘한 문법은 그냥 pass 
# 앞의 위치에 있는 파일을 x_train으로 쓴다

np.save('./data/mnist_train_x.npy', arr = x_train) 
np.save('./data/mnist_test_x.npy', arr = x_test)
np.save('./data/mnist_train_y.npy', arr = y_train)
np.save('./data/mnist_test_y.npy', arr = y_test)


# # 데이터 전처리 1. 원핫인코딩
# # y data
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)

# # 데이터 전처리 2. 정규화
# # x data
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

# print(x_train.shape)



