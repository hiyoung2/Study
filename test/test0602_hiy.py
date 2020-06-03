import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM
from keras.layers.merge import concatenate 
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# 1. 데이터 준비
hite = pd.read_csv("./data/csv/hite.csv",
                   index_col = 0, header = 0, encoding = 'cp949', sep = (','))

ss = pd.read_csv("./data/csv/samsung.csv",
                  index_col = 0, header = 0, encoding='cp949', sep = (','))

hite.info()

# Nan 제거하기

# hite = hite.dropna(how = 'all') # nan으로 채워진 모든 행들 삭제
hite = hite.dropna(axis = 0) # 위와 같음

# Nan 채우기
hite = hite.fillna(method='bfill')

print(hite)

'''
hite = hite.sort_values(['일자'], ascending = [True])
ss = ss.sort_values(['일자'], ascending = [True])

print(hite)
print(ss)

for i in range(len(hite.index)) :
    for j in range(len(hite.iloc[i])) :
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',', ''))

# print(len(hite.iloc[0]))
print(hite)

ss = ss.dropna(how = 'all')
print(ss)

for i in range(len(ss.index)) :
    ss.iloc[i,0] = int(ss.iloc[i,0].replace(',', ''))

print(ss)


'''

'''
# 1-1. 데이터 npy로 저장
a = hite.values
b = ss.values

np.save('./data/hite.npy', arr = a)
np.save('./data/samsung.npy', arr = b)
'''
'''
# 1-2. 데이터 준비
hdata = np.load('./data/hite.npy', allow_pickle=True)
sdata= np.load('./data/samsung.npy', allow_pickle=True)

print(hdata)
print(sdata)


print('hdat.shape : ', hdata.shape) # (509, 5)
print('sdata.shape : ', sdata.shape) # (509, 1)

hdata = hdata[:, :-1] # 거래량은 제거
print(hdata)
print('hdata_new : ', hdata) # (509, 4)


def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(hdata, 5, 1)
x2, y2 = split_xy(sdata, 5, 1)

print("=======================")
print(x1[0, :], "\n", y1[0])

print('x1_shape : ', x1.shape)
print('y1_shape : ', y1.shape)

'''

# 1-3. 데이터 전처리
# PCA

# MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x1_h)
# x1_h = scaler.transform(x1_h)
# print('scaled_x1_h : ', x1_h)

# scaler.fit(x2_s)
# x2_s = scaler.transform(x2_s)
# print('scaled_x2_s : ', x2_s)


# 1-4. train, test split



# 2. 모델 구성

# 3. 컴파일, 훈련

# 4. 평가, 예측
