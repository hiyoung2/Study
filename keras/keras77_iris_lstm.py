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

print(x)
print(y)

# 1.1 데이터 전처리
y = np_utils.to_categorical(y)


scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print('x_transform : ', x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size  = 0.8
)

print(x_train.shape) 
print(x_test.shape)  
print(y_train.shape) 
print(y_test.shape)  
