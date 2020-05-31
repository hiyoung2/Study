import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. 데이터 준비
diabetes = load_diabetes()
x = diabetes['data']
y = diabetes['target']

print(x)
print(y)

print('x.shape : ', x.shape) # (442, 10)
print('y.shape : ', y.shape) # (442,)

# 1.1 데이터 전처리 - Scaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# 1.2 데이터 분리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y , train_size = 0.8, random_state = 77, shuffle = True
)

print('x_train.shape : ', x_train.shape) # (353, 10)
print('x_test.shape : ', x_test.shape)   # (89, 10)

# 1.3 데이터 shape 맞추기

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], 1)


# x_train = x_train.reshape(x_train.shape[0], 5, 2)
# x_test = x_test.reshape(x_test.shape[0], 5, 2)


# 2. 모델 구성
model = Sequential()

model.add(LSTM(50, input_shape = (10, 1)))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(700))
model.add(Dense(800))
model.add(Dense(900))
model.add(Dense(1000))

model.add(Dense(5000))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
modelpath = './model/{epoch:02d}--{loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 32)

print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)
# print(y_pred)

# RMSE, R2

from sklearn.metrics import mean_squared_error
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('RSME : ', rmse(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print('R2 : ', r2)


# 시각화
plt.figure(figsize =(10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c ='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['mse'], marker = '*', c = 'green', label = 'mse')
plt.plot(hist.history['val_mse'], marker = '*', c = 'purple', label = 'val_mse')
plt.grid()
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

'''
쓰레기 수집가

model.add(LSTM(100, input_shape = (10, 1)))
model.add(Dense(500))
model.add(Dense(800))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(1))

epo = 100, batch = 1
RSME :  60.627076009353296
R2 :  0.4175919854431587

epo = 200, batch = 32
RSME :  60.26515608533354
R2 :  0.42452472658875284
'''
'''
model.add(LSTM(50, input_shape = (10, 1)))
# model.add(LSTM(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(10))
model.add(Dense(1))

epo = 200, batch = 32
RSME :  62.654890766667506
R2 :  0.3779804269257999
'''

'''
model.add(LSTM(50, input_shape = (10, 1)))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1))

epo = 200, batch = 32
RSME :  60.29856363137002
R2 :  0.42388652877704047
'''

'''
model.add(LSTM(50, input_shape = (10, 1)))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(2000))
model.add(Dense(10))
model.add(Dense(1))

epo = 200, batch = 32
RSME :  59.003896868823546
R2 :  0.4483603338037586
'''
'''
model.add(LSTM(50, input_shape = (10, 1)))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(700))
model.add(Dense(900))

model.add(Dense(2000))
model.add(Dense(10))
model.add(Dense(1))

epo = 200, batch = 32
RSME :  57.381510216486355
R2 :  0.47827932811123075
'''

'''
model.add(LSTM(50, input_shape = (10, 1)))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(700))
model.add(Dense(900))
model.add(Dense(1000))

model.add(Dense(2000))
model.add(Dense(10))
model.add(Dense(1))

epo 200, batch = 32
RSME :  56.62241076511971
R2 :  0.4919916993497121
'''