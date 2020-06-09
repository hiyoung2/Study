import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submit = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", data.shape)     # (10000, 75)    
print("test.shape : ", x_pred.shape)    # (10000, 71)         
print("submit.shape : ", submit.shape)  # (10000, 4)

# 결측치 확인 및 처리
# 각 column 별로 결측치가 얼마나 있는지 알 수 있다
print(data.isnull().sum()) 

# 선형보간법 적용(모든 결측치가 처리 되는 건 아니기 때문에 검사가 필요하다)
data = data.interpolate() 
x_pred = x_pred.interpolate()

# 결측치에 평균을 대입
data = data.fillna(data.mean())
x_pred = x_pred.fillna(x_pred.mean())

# 결측치 모두 처리 됨을 확인
# print(data.isnull().sum()) 
# print(x_pred.isnull().sum()) 


np.save("./data/dacon/comp1/data.npy", arr = data)
np.save("./data/dacon/comp1/x_pred.npy", arr = x_pred)


data = np.load("./data/dacon/comp1/data.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/x_pred.npy", allow_pickle = True)

print("data.shape :", data.shape)     # (10000, 75)
print("x_pred.shape :", x_pred.shape) # (10000, 71)


x = data[:, :71]
y = data[:, -4:]

print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)
print()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

print("x_train.shape :", x_train.shape) # (8000, 71)
print("x_test.shape :", x_test.shape)   # (2000, 71)
print()

# scaler 사용시 주의 : 2차원으로 넣어야 한다
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# LSTM 모델 : 3차원, 따라서 reshape!
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


print("====x_train, x_test shape 확인====")
print("x_train.re :", x_train.shape) # (8000, 71, 1)
print("x_test.re :", x_test.shape)   # (2000, 71, 1)



# 2. 모델 구성

input1 = Input(shape = (71, 1))
dense1 = LSTM(10, activation = 'relu')(input1)
dense1 = Dense(50, activation = 'relu')(dense1) 
dense1 = Dropout(0.1)(dense1) 
dense1 = Dense(30, activation = 'relu')(dense1) 
output1 = Dense(4)(dense1)

model = Model(inputs = input1, outputs=output1)

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32, callbacks = [es], validation_split = 0.2, verbose = 1)

# 4. 평가, 에측
loss, mae = model.evaluate(x_test, y_test, batch_size = 32)
print("loss :", loss)
print("mae :", mae)

# x_pred도 LSTM 3차원 모델에 맞게 reshape 해줘야 한다!
# 안 하면 y_pred 구하는 과정에서 계속 와꾸 에러가 발생함

x_pred = x_pred.reshape(10000, 71, 1)

y_pred = model.predict(x_pred)
print("y_pred :", y_pred)

# csv 파일 만들기(submit 파일)
# y_pred.to_csv(경로)

from pandas import DataFrame

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv("./dacon/comp1/submit_LSTM.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
