import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input


# cvs 파일 읽기
x_data = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
y_data = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp3/test_features.csv", header = 0, index_col = 0)
y_pred = pd.read_csv("./data/dacon/comp3/sample_submission.csv", header = 0, index_col = 0)

# npy 변환, 저장
np.save("./data/dacon/comp3/x_data.npy", arr = x_data)
np.save("./data/dacon/comp3/y_data.npy", arr = y_data)
np.save("./data/dacon/comp3/x_pred.npy", arr = x_pred)


# npy 불러오기
x_data = np.load("./data/dacon/comp3/x_data.npy", allow_pickle = True)
y_data = np.load("./data/dacon/comp3/y_data.npy", allow_pickle = True)
x_pred = np.load("./data/dacon/comp3/x_pred.npy", allow_pickle = True)

# 데이터 shape 확인
print("x_data.shape :", x_data.shape) # (1050000, 5)
print("y_data.shape :", y_data.shape) # (2800, 4)
print("x_pred.shape :", x_pred.shape) # (262500, 5)
print()

# 데이터 슬라이싱

print("=====데이터 슬라이싱=====")
x_data = x_data[:, 1:]
print("x_data_sl :", x_data.shape) # (1050000, 4)(time column 제거)

x_pred = x_pred[:, 1:]
print("x_pred_sl :", x_pred.shape) # (262500, 4)

print()

# scaler 사용
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_pred = scaler.transform(x_pred)


# DNN 모델 위해 reshape
x_data = x_data.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2
)

# 2. 모델 구성


input1 = Input(shape = (375*4, ))
dense1 = Dense(30, activation = 'relu')(input1)
dense1 = Dense(50, activation = 'relu')(input1)
dense1 = Dense(70, activation = 'relu')(input1)
dense1 = Dense(90, activation = 'relu')(input1)
dense1 = Dense(110, activation = 'relu')(input1)
dense1 = Dense(50, activation = 'relu')(input1)
dense1 = Dense(10, activation = 'relu')(input1)
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
y_pred.to_csv("./dacon/comp3/submit_dnn.csv", header = ["X", "Y", "M", "V"], index = True, index_label="id" )
