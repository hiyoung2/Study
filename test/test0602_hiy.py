import numpy as np
import pandas as pd
import numpy as np
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

samsung = pd.read_csv("./data/csv/samsung.csv",
                  index_col = 0, header = 0, encoding='cp949', sep = (','))
                 
hitedata = hite.iloc[:509]
print(hitedata)

ssdata = samsung.iloc[:509]
print(ssdata)

hitedata = hitedata.replace(np.nan, 0)
print(hitedata)

ssdata = ssdata.replace(np.nan, 0)
print(ssdata)

# hitedata.astype({'시가':'int', '고가':'int', '저가':'int', '종가':'int', '거래량':'int'}).dtypes
# print(hitedata.dtypes)

# for i in range(len(hitedata.index)) :
#     for j in range(len(hitedata.iloc[i])) :
#         hitedata.iloc[i, j] = int(hitedata.iloc[i, j].replace(',', ''))

# print(hitedata)

# # 1-1. 데이터 npy로 저장
# a = hitedata.values
# b = ssdata.values

# np.save('./data/hite.npy', arr = a)
# np.save('./data/samsung.npy', arr = b)

# # 1-2. 데이터 준비
# x = np.load('./data/hite.npy', allow_pickle=True)
# y = np.load('./data/samsung.npy', allow_pickle=True)

# print(x)
# print(y)
#############################################################
# print(hite.dtypes)
# hite.astype({'시가':'int', '고가':'int', '저가':'int', '종가':'int', '거래량':'int'}).dtypes
# print(hite.dtypes)

##############################################################

##############################################################
# print(hite.dtypes)
# hite.astype({'시가':'int', '고가':'int', '저가':'int', '종가':'int', '거래량':'int'}).dtypes
# print(hite.dtypes)
##############################################################

# hite = np.asfarray(hite, float)
# print(hite)
#############################################################

# hite = np.asarray(hite, dtype = np.float32, order = 'C')
# print(hite)
#############################################################
# hitedata = hite.astype(np.float)
# print(hitedata)
#############################################################
# hitedata = hite.values

# hitedata[:, :] = float(hitedata[:, :])
# print(type(hitedata[:,:]))
####################
# hitedata = hite.str.replace(",").astype(float)
# print(hitedata)
########

# a[:, 0].astype(int)
# print(a)



# 1-3. 데이터 전처리
# MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)

# print(x)


# 1-4. x1, x2, y1


# 1-5. train, test split


# 2. 모델 구성

# 3. 컴파일, 훈련

# 4. 평가, 예측
