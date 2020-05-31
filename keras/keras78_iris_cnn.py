# keras78_iris_cnn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.decomposition import PCA

# 1. 데이터 준비
iris = load_iris()
x = iris['data']
y = iris['target']

print(x)
print(y)

print('x.shape : ', x.shape) # (150, 4)
print('y.shape ; ', y.shape) # (150,)

# 1.1 데이터 전처리
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (150, 3)

# 1.2 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print('x_scaled : ', x.shape) # (150, 4)

# column이 4개밖에 안 되므로 안 하는 게 좋을 것 같지만 연습삼아 시도
# pca = PCA(n_components = 2)
# pca.fit(x)
# x_pca = pca.transform(x)
# print(x_pca.shape) # (150, 2)


# 1.3 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

# 1.4 데이터 shape 맞추기
x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)

# 2. 모델 구성
model = Sequential()

model.add(Conv2D(50, (2, 2), input_shape = (2, 2, 1)))
model.add(Conv2D(70, (2, 2), padding = 'same'))
model.add(Flatten())
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# # 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
modelpath = './model/{epoch:02d}--{acc:.4f}.hdf5' # hdf의 d와 f는 02d와 4f의 df?
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'acc', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, callbacks = [checkpoint], verbose = 1)

# 4. 평가, 예측
loss, acc  = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))

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
plt.plot(hist.history['acc'], marker = '*', c = 'green', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '*', c = 'purple', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

'''
loss :  0.15457082632152203
acc :  1.0
'''
'''
model.add(Conv2D(11, (2, 2), input_shape = (2, 2, 1)))
model.add(Conv2D(22, (2, 2), padding = 'same'))
model.add(Conv2D(44, (2, 2), padding = 'same'))
model.add(Conv2D(66, (2, 2), padding = 'same'))
model.add(Conv2D(99, (2, 2), padding = 'same'))
model.add(Conv2D(111, (2, 2), padding = 'same'))
model.add(Conv2D(88, (2, 2), padding = 'same'))
model.add(Conv2D(55, (2, 2), padding = 'same'))
model.add(Conv2D(33, (2, 2), padding = 'same'))
model.add(Conv2D(11, (2, 2), padding = 'same'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch = 1
loss :  0.11775056918268093
acc :  0.9333333373069763
'''

'''
model.add(Conv2D(11, (2, 2), input_shape = (2, 2, 1)))
model.add(Conv2D(22, (2, 2), padding = 'same'))
model.add(Conv2D(44, (2, 2), padding = 'same'))
model.add(Conv2D(66, (2, 2), padding = 'same'))
model.add(Conv2D(99, (2, 2), padding = 'same'))
model.add(Conv2D(111, (2, 2), padding = 'same'))
model.add(Conv2D(88, (2, 2), padding = 'same'))
model.add(Conv2D(55, (2, 2), padding = 'same'))
model.add(Conv2D(33, (2, 2), padding = 'same'))
model.add(Conv2D(11, (2, 2), padding = 'same'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch 32
loss :  0.06772632348533184
acc :  0.9666666388511658
'''
'''
model.add(Conv2D(50, (2, 2), input_shape = (2, 2, 1)))
model.add(Conv2D(70, (2, 2), padding = 'same'))
model.add(Flatten())
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch = 32
loss :  0.01219688486950948
acc :  1.0
'''