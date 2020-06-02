# (행, 13)
# boston  유명한 예제 데이터셋, 기초예제
# 케라스에 있는 예제 다 해서 싸이킷런에서 불러옴
# PCA

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# 케라스랑 다르게 데이터 불러옴
# '''
# data       : x값
# target     : y값
# '''
import numpy as np
from sklearn.datasets import load_boston
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

dataset = load_boston()
x = dataset['data']
y = dataset['target']

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape)   # x_train.shape :  (404, 13)
print("x_test.shape : ", x_test.shape)     # x_test.shape :  (102, 13)
print("y_train.shape : ", y_train.shape)   # y_train.shape :  (404,)
print("y_test.shape : ", y_test.shape)     # y_test.shape :  (102,)

# 1. 데이터 준비

# 1.2 데이터 전처리 (x)
scaler = StandardScaler()

scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape) # (404, 13)

# x_train = x_train.reshape(x_train.shape[0], x.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x.shape[1], 1)

x_train = x_train.reshape(x_train.shape[0], 1, 13)
x_test = x_test.reshape(x_test.shape[0], 1, 13)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(50, return_sequences = True,  input_shape = (1, 13), activation = 'relu'))
model.add(LSTM(70, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping

# es = EarlyStopping(monitor  = 'loss', patience = 30, mode = 'auto')

# modelpath = './model/{epoch:02d}-{loss:.4f}.hdf5'

# checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', 
                            #  save_best_only=True, mode = 'auto') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

# hist = model.fit(x_train, y_train, epochs = 200, batch_size = 10, validation_split = 0.2, callbacks = [es, checkpoint], verbose = 1)

hist = model.fit(x_train, y_train, epochs = 200, batch_size = 10, validation_split = 0.2, verbose = 1)

# hist = model.fit(x_train, y_train, epochs = 250, batch_size = 10, verbose = 1)

'''
# 3.1 시각화
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
plt.grid() # 모눈종이처럼 그림에 가로 세로 줄이 그어져 나옴
plt.title('loss')      
plt.ylabel('loss')      
plt.xlabel('epoch')          
# plt.legend(['loss', 'val_loss']) 
plt.legend(loc = 'upper right') # loc : location, 우측 상단

plt.subplot(2, 1, 2) # 2행 1열의 2번째 그림
plt.plot(hist.history['mse'], marker = '.', c = 'green', label = 'mse')
plt.plot(hist.history['val_mse'], marker = '.', c = 'purple', label = 'val_mse')
plt.grid() 
plt.title('mse')      
plt.ylabel('mse')      
plt.xlabel('epoch')          
# plt.legend(['acc', 'val_acc']) 
plt.legend(loc = 'upper right')
# plt.show()  
'''
# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 10)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)
# print(y_pred)

from sklearn.metrics import mean_squared_error
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", rmse(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

'''
1차
model.add(LSTM(11, input_shape = (13, 1), activation = 'relu'))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
epo 500, patience 50, val_split 0.3, batch 10
stop : 448, best : 398
RMSE :  4.019755179678341
R2 :  0.78351150349943
'''
'''
2차
model.add(LSTM(11, return_sequences = True,  input_shape = (13, 1), activation = 'relu'))
model.add(LSTM(22))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
epo 400, es x, batch 10, val 0.3, best : 375
RMSE :  3.9620766543537767
R2 :  0.814195001352444
'''
'''
3차
model.add(LSTM(11, return_sequences = True,  input_shape = (13, 1), activation = 'relu'))
model.add(LSTM(22, activation = 'relu'))
model.add(Dense(55, activation = 'relu'))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
epo 375, es x, batch 10, val 0.3, best : 368
RMSE :  4.557340681997267
R2 :  0.7758861448553874

'''

'''
model.add(LSTM(11, return_sequences = True,  input_shape = (1, 13), activation = 'relu'))
model.add(LSTM(22))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
epo 200, batch 32, es = 20, val = 0.3
RMSE :  3.055696874823824
R2 :  0.881298347600631
'''

'''
model.add(LSTM(20, return_sequences = True,  input_shape = (1, 13), activation = 'relu'))
model.add(LSTM(40))
model.add(Dense(55))
model.add(Dense(77))
model.add(Dropout(0.2))
model.add(Dense(99))
model.add(Dropout(0.3))
model.add(Dense(33))
model.add(Dense(1))

epo = 200, batch = 10, es = 20, val = 0.2
RMSE :  3.389819670414019
R2 :  0.8834626115364802
'''

'''
model.add(LSTM(50, return_sequences = True,  input_shape = (1, 13), activation = 'relu'))
# model.add(Dropout(0.2))
model.add(LSTM(70))
# model.add(Dropout(0.3))
# model.add(Dense(90))
# model.add(Dense(110))
# model.add(Dropout(0.3))
# model.add(Dense(130))
# model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dense(1))

epo 250, batch = 10, val = 0.3
RMSE :  2.372629411176336
R2 :  0.9324220721549005
'''