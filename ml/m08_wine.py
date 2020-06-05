# 0605 day 20(4주차,,,)

# 머신러닝 기법

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('./data/csv/winequality-white.csv',
                    index_col = None, header = 0, sep = ';' )

print(wine.head())
print(wine.tail())

print('wine.shape : ', wine.shape) # (4898, 12)

print(type(wine)) # <class 'pandas.core.frame.DataFrame'>

wine = wine.values
print(type(wine)) # <class 'numpy.ndarray'>

np.save('./data/wine.npy', arr = wine)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 데이터 준비
x_data = wine[:, 0:11]
y_data = wine[:, 11]

print(x_data)
print(y_data)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, random_state = 11, shuffle = True
)

print('x_train.shape  : ', x_train.shape) # (3918, 11)
print('y_train.shape : ', y_train.shape)  # (3918,)

# scaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# model = SVC()
'''
SCORE :  0.5857142857142857
ACC :  0.5857142857142857
'''

# model = KNeighborsClassifier(n_neighbors = 1)
'''
SCORE :  0.6632653061224489
ACC :  0.6632653061224489

'''

model = RandomForestClassifier()
'''
SCORE :  0.6989795918367347
ACC :  0.6989795918367347
'''

# 3. 훈련
model.fit(x_train, y_train)
socre = model.score(x_test, y_test)

# 4. 평가, 예측
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("SCORE : ", socre)
print("ACC : ", acc)






