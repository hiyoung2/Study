# keras82_breast_lstm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.decomposition import PCA

# 1. 데이터 준비
bc = load_breast_cancer()
x = bc['data']
y = bc['target']

print(x)
print(y)

print('x.shape : ', x.shape)  # (569, 30)
print('y.shape : ', y.shape)  # (569, )

# 1.1 데이터 전처리(One Hot Encoder)
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (569, 2)

# 1.2 데이터 전처리(Scaler)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# 1.3 데이터 전처리(PCA)
pca = PCA(n_components= 10)
pca.fit(x)
x_pca = pca.transform(x)

# print('x_pca.shape : ', x_pca.shape) # x_pca.shape :  (569, 10)

# 1.4 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 77, shuffle = True
)

print('x_train.shape : ', x_train.shape) # x_train.shape :  (455, 30)
print('x_test.shape : ' , x_test.shape)  # x_test.shape :  (114, 30)

# x_train, x_test, y_train, y_test = train_test_split(
#     x_pca, y, train_size = 0.8, random_states = 77, shuffle = True
# )

# print('x_train_pca.shape : ', x_train.shape) # x_train.shape :  (455, 10)
# print('x_test_pca.shape : ' , x_test.shape)  # x_test.shape :  (114, 10)

# 1.5 데이터 모양 맞추기

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델 구성

model = Sequential()

model.add(LSTM(50, input_shape = (30, 1)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

modelpath = './model/{epoch:02d}--{loss:.4fd}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, callbacks = [es, checkpoint], verbose = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y, axis = 1))

print('loss : ', loss)
print('acc : ', acc)

'''
model.add(LSTM(50, input_shape = (30, 1)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

epo = 100, batch = 32, val = 0.2, es = x
loss :  0.07831162227350369
acc :  0.9736841917037964
'''