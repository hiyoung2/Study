# comp3 / dacon / randomizedsearchCV + pipe + dnn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

x_data = pd.read_csv('./data/dacon/comp3/train_features.csv', header = 0, index_col = 0)
y_data = pd.read_csv('./data/dacon/comp3/train_target.csv', header = 0, index_col= 0)
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv', header = 0, index_col = 0)
y_pred = pd.read_csv('./data/dacon/comp3/sample_submission.csv', header = 0, index_col = 0)

# 넘파이로 파일형식 변환 저장
np.save('./data/dacon/comp3/x_data.npy', arr = x_data)
np.save('./data/dacon/comp3/y_data.npy', arr = y_data)
np.save('./data/dacon/comp3/x_pred.npy', arr = x_pred)

# 넘파이 데이터 불러오기
x_data = np.load('./data/dacon/comp3/x_data.npy', allow_pickle = True)
y_data = np.load('./data/dacon/comp3/y_data.npy', allow_pickle = True)
x_pred = np.load('./data/dacon/comp3/x_pred.npy', allow_pickle = True)

# 데이터 구조 파악
print()
print("x_data.shape :", x_data.shape) # (1050000, 5)
print("y_data.shape :", y_data.shape) # (2800, 4)
print("x_pred.shape :", x_pred.shape) # (262500, 5)
print()

# 데이터 슬라이싱
x_data = x_data[:, 1:]
x_pred = x_pred[:, 1:]

print()
print("x.shape :", x_data.shape)           # (1050000, 4)
print("x_pred.shape :", x_pred.shape) # (262500, 4)
print()

# 스케일링
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_pred = scaler.fit_transform(x_pred)
# print()
# print(x)
# print()

# 데이터 reshape
x_data = x_data.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)
print()
print("x.reshape :", x_data.reshape)
print("x_pred.reshape :", x_pred.reshape)
print()

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2
)

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape) 
print("y_train.shape :", y_train.shape) 
print("y_test.shape :", y_test.shape) 



# 2. 모델 구성


