import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)

test = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)

submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", train.shape)             # (10000, 75) : x_train, x_test로 만들어야 함
print("test.shape : ", test.shape)               # (10000, 71) : x_pred
print("submission.shape : ", submission.shape)   # (10000, 4)  : y_pred

# 결측치

print(train.isnull().sum()) 

train = train.interpolate() 

test = test.interpolate()

# print(train.head())
train = train.fillna(method = 'bfill')
print(train.head())


train = train.values
test = test.values
submission = submission.values

print(type(train))

np.save("./data/dacon/comp1/train.npy", arr = train)
np.save("./data/dacon/comp1/test.npy", arr = test)
np.save("./data/dacon/comp1/submission.npy", arr = submission)


data = np.load("./data/dacon/comp1/train.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/test.npy", allow_pickle = True)
y_pred = np.load("./data/dacon/comp1/submission.npy", allow_pickle = True)

print("data.shape :", data.shape)
print("x_pred.shape :", x_pred.shape)
print("y_pred.shape :", y_pred.shape)

x = data[:, :71]
y = data[:, -4:]

print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

print("x_train.shape :", x_train.shape) # (8000, 71)
print("x_test.shape :", x_test.shape)   # (2000, 71)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print("x_train.re :", x_train.shape) # (8000, 71, 1)
print("x_test.re :", x_test.shape)   # (2000, 71, 1)




# 2. 모델 구성

input1 = Input(shape = (71, 1))
dense1 = LSTM(77, activation = 'relu')(input1)
dense1 = Dropout(0.5)(dense1) 
dense1 = Dense(99, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(111, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(133, activation = 'relu')(dense1) 
dense1 = Dropout(0.5)(dense1) 
dense1 = Dense(155, activation = 'relu')(dense1) 
dense1 = Dropout(0.5)(dense1) 
dense1 = Dense(88, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(66, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(55, activation = 'relu')(dense1) 
dense1 = Dropout(0.1)(dense1) 
output1 = Dense(4)(dense1)

model = Model(inputs = input1, outputs=output1)

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 200, batch_size = 32, callbacks = [es], validation_split = 0.2, verbose = 1)

# 4. 평가, 에측
loss, mae = model.evaluate(x_test, y_test, batch_size = 32)
print("loss :", loss)
print("mae :", mae)

x_pred = x_pred.reshape(10000, 71, 1)

y_pred = model.predict(x_pred)
print("y_pred :", y_pred)

# csv 파일 만들기(submit 파일)
# y_pred.to_csv(경로)

from pandas import DataFrame

a = np.arange(10000,20000)
submission = y_pred
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
