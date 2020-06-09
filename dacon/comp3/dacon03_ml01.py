import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras.layers import Input

x_data = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
y_data = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp3/test_features.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp3/sample_submission.csv", header = 0, index_col = 0)

# shape 확인
print("x_data.shape :", x_data.shape)    # (1050000, 5)
print("y_data.shape :", y_data.shape)    # (2800, 4)
print("x_pred.shape :", x_pred.shape)    # (262500, 5)
print("submission :", submission.shape)  # (700, 4)

# 결측치 확인 및 처리
print(x_data.isnull().sum())
print()
print(y_data.isnull().sum())

'''
Time    0
S1      0
S2      0
S3      0
S4      0
dtype: int64

X    0
Y    0
M    0
V    0
dtype: int64
'''

# 1. 데이터 준비

np.save("./data/dacon/comp3/x_data.npy", arr = x_data)
np.save("./data/dacon/comp3/y_data.npy", arr = y_data)
np.save("./data/dacon/comp3/x_pred.npy", arr = x_pred)

x_data = np.load("./data/dacon/comp3/x_data.npy", allow_pickle = True)
y_data = np.load("./data/dacon/comp3/y_data.npy", allow_pickle = True)
x_pred = np.load("./data/dacon/comp3/x_pred.npy", allow_pickle = True)

# 데이터 슬라이싱
# 현재 x_data, x_pred data에 time 이라는 컬럼은 실제 데이터에 포함 되지 않는다고 본다
x_data = x_data[:, 1:] # index_col = 0으로 설정헀으므로 [0]번째 컬럼은 time이 되므로 [1]번째 컬럼부터 x_data로 사용
x_pred = x_pred[:, 1:]

print("데이터 슬라이싱")
print("x_data :", x_data.shape)
print("x_pred :", x_pred.shape)

# train_test_split

x_data = x_data.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2
)

print("x_train.shape :", x_train.shape)  # (2240, 1500)
print("x_test.shape :", x_test.shape)    # (560, 1500)
print("y_train.shape :", y_train.shape)  # (2240, 4)
print("y_test.shape :", y_test.shape)    # (560, 4)


# 2. 모델 구성 (머신러닝)

pipe = Pipeline([("scaler", StandardScaler()), ('ensemble', RandomForestRegressor())])

# 3. 훈련
pipe.fit(x_train, y_train)

# 4. 평가, 예측
loss = pipe.score(x_test, y_test)

y_pred = pipe.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

submit = pipe.predict(x_pred)

print("loss :", loss)
print("mse :", mse)
print("submit : ", submit[:5, :])

# 최종 파일 변환
a = np.arange(2800, 3500)

submit = pd.DataFrame(submit, a)
submit.to_csv("./data/dacon/comp3/submit1.csv", header = ["X", "Y", "M", "V"], index = True, index_label = "id")