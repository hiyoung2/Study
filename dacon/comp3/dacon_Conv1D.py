import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_data = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)

y_data = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)

x_pred = pd.read_csv("./data/dacon/comp3/test_features.csv", header = 0, index_col = 0)

submit = pd.read_csv("./data/dacon/comp3/sample_submission.csv", header = 0, index_col = 0)


# 결측치 확인 및 처리
print(x_data.isnull().sum())
print()
print(y_data.isnull().sum())


# npy 로 변환, 저장
np.save("./data/dacon/comp3/x_data.npy", arr = x_data)
np.save("./data/dacon/comp3/y_data.npy", arr = y_data)
np.save("./data/dacon/comp3/x_pred.npy", arr = x_pred)

# npy 파일 불러오기
x_data = np.load("./data/dacon/comp3/x_data.npy", allow_pickle = True)
y_data = np.load("./data/dacon/comp3/y_data.npy", allow_pickle = True)
x_pred = np.load("./data/dacon/comp3/x_pred.npy", allow_pickle = True)


print("x_data.shape :", x_data.shape) # (1050000, 5)
print("y_data.shape :", y_data.shape) # (2800, 4)
print("x_pred.shape :", x_pred.shape) # (262500, 5)
print()

print("=====데이터 슬라이싱=====")
x_data = x_data[:, 1:]
print("x_data:", x_data.shape) # (1050000, 4)(time column 제거)

x_pred = x_pred[:, 1:]
print("x_pred :", x_pred.shape) # (262500, 4)

print()

# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_pred = scaler.transform(x_pred)


# Conv1D 모델링 하기 위해 reshape 
# id 하나당 375개 있음
x_data = x_data.reshape(2800, 375, 4)
x_pred = x_pred.reshape(700, 375, 4)


# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2
)

print("=====train_test_split=====")
print("x_train.shape :", x_train.shape)
print("x_train.shape :", x_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape :", y_test.shape)


# 2. 모델 구성
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Conv1D, Flatten

input1 = Input(shape = (375, 4))
dense1 = Conv1D(70, 2 , activation = 'relu')(input1)
dense1 = Dropout(0.3)(dense1) 
dense1 = Flatten()(dense1)
dense1 = Dense(90, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(110, activation = 'relu')(dense1) 
dense1 = Dropout(0.4)(dense1) 
dense1 = Dense(130, activation = 'relu')(dense1) 
dense1 = Dropout(0.4)(dense1) 
dense1 = Dense(150, activation = 'relu')(dense1) 
dense1 = Dropout(0.5)(dense1) 
dense1 = Dense(170, activation = 'relu')(dense1) 
dense1 = Dropout(0.5)(dense1) 
dense1 = Dense(140, activation = 'relu')(dense1) 
dense1 = Dropout(0.4)(dense1) 
dense1 = Dense(80, activation = 'relu')(dense1) 
dense1 = Dropout(0.7)(dense1) 
dense1 = Dense(60, activation = 'relu')(dense1) 
dense1 = Dropout(0.3)(dense1) 
dense1 = Dense(50, activation = 'relu')(dense1) 
dense1 = Dropout(0.1)(dense1) 
dense1 = Dense(30, activation = 'relu')(dense1) 
dense1 = Dropout(0.1)(dense1) 
output1 = Dense(4)(dense1)

model = Model(inputs = input1, outputs=output1)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 32)

y_pred = model.predict(x_pred)

print("loss :", loss)
print("mse :", mse)
print("y_pred : ", y_pred)


a = np.arange(2800, 3500)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv("./dacon/comp3/submit_Conv1D.csv", header = ["X", "Y", "M", "V"], index = True, index_label="id" )
