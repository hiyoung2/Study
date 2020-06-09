import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Input, Dropout
from keras.layers import Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling2D
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# dacon_comp1 데이터 불러오기

train = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
test = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submit = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", train.shape)             # (10000, 75) : x_train, x_test로 만들어야 함
print("test.shape : ", test.shape)               # (10000, 71) : x_pred
print("submission.shape : ", submit.shape)   # (10000, 4)  : y_pred


# 결측치 확인
print(train.isnull().sum()) 

train = train.interpolate() 

test = test.interpolate()
# print(train.head())
# train = train.fillna(train.mean())
# print(train.head())

# 결측치보완(이전 값 대입)
# print(train.head())
train = train.fillna(method ='bfill')
print(train.head())

test = test.fillna(method = 'bfill')

# 결측치보완(평균값 대입법)
# print(train.head())
# train = train.fillna(train.mean())
# print(train.head())

np.save("./data/dacon/comp1/train.npy", arr = train)
np.save("./data/dacon/comp1/test.npy", arr = test)

data = np.load("./data/dacon/comp1/train.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/test.npy", allow_pickle = True)

x = data[:, :71]
y = data[:, -4:]


print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

print("x_train.shape :", x_train.shape)  # (8000, 71)
print("x_test.shape :", x_test.shape)    # (2000, 71)
print("y_train.shape :", y_train.shape)  # (8000, 4)
print("y_test.shape :", y_test.shape)    # (2000, 4)


print("x_pred.shape :", x_pred.shape) # (10000, 71)



# kfold = KFold(n_splits = 5, shuffle = True)

pipe = Pipeline([("scaler", StandardScaler()), ('ensemble', RandomForestRegressor())])

pipe.fit(x_train, y_train)

loss = pipe.score(x_test,y_test)

y_pred = pipe.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)


submit = pipe.predict(x_pred)


print("loss :", loss)
print("mae :", mae)


a = np.arange(10000,20000)
submit= pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp1/submit_RFR.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
