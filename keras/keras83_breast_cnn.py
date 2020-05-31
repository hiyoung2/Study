# keras83_breast_cnn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
print('x.shape : ', x.shape)

# 1.3 데이터 전처리(PCA)
# pca = PCA(n_components= 25)
# pca.fit(x)
# x_pca = pca.transform(x)

# print('x_pca.shape : ', x_pca.shape) # x_pca.shape :  (569, 25)

# 1.4 데이터 분리

x_train, x_test, y_train, y_test = train_test_split(
    x , y, train_size = 0.8, random_state = 77, shuffle = True
)

print('x_train.shape : ', x_train.shape)
print('x_test.shape : ' , x_test.shape)

# x_train, x_test, y_train, y_test = train_test_split(
#     x_pca, y, train_size = 0.8, random_state = 77, shuffle = True
# )

# print('x_pca_train.shape : ', x_train.shape) # x_train.shape :  (455, 25)
# print('x_pca_test.shape : ' , x_test.shape)  # x_test.shape :  (114, 25)

# 1.5 데이터 모양 맞추기

x_train = x_train.reshape(x_train.shape[0], 6, 5, 1)
x_test = x_test.reshape(x_test.shape[0], 6, 5, 1)

# x_train = x_train.reshape(x_train.shape[0], 5, 5, 1)
# x_test = x_test.reshape(x_test.shape[0], 5, 5, 1)

print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)


# 2. 모델 구성
model = Sequential()

model.add(Conv2D(10, (2, 2), input_shape = (6, 5, 1)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(130))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(20))
model.add(Dense(2, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

modelpath = './model/{epoch:02d}--{loss:.4fd}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split = 0.2, callbacks = [es, checkpoint], verbose = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 10)

y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y, axis = 1))

print('loss : ', loss)
print('acc : ', acc)

'''
loss :  0.21596390517005484
acc :  0.9736841917037964
'''
'''
model.add(Conv2D(10, (2, 2), input_shape = (5, 5, 1)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(130))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(20))
model.add(Dense(2, activation = 'sigmoid'))

epo = 100, batch= 32, val = 0.2, pca 적용
loss :  0.8780053765104529
acc :  0.9561403393745422
'''