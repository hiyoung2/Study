import numpy as np  
import matplotlib.pyplot as plt

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# 1. 데이터 준비

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(y_test.shape)  # (10000, 1)

# 1.1 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 1.2 데이터 전처리
x_train = x_train.reshape(50000, 32*32*3).astype('float32') / 255.0
x_test = x_test.reshape(10000, 32*32*3).astype('float32') / 255.0
print(x_train.shape)

# 2. 모델 구성

input1 = Input(shape = (32*32*3, ))
dense1 = Dense(99, activation = 'relu')(input1) 
dense2 = Dropout(0.2)(dense1)
dense3 = Dense(166)(dense2) 
dense4 = Dropout(0.3)(dense3)
dense5 = Dense(252, activation = 'relu')(dense4) 
dense6 = Dropout(0.4)(dense5) 
dense7 = Dense(177)(dense6) 
dense8 = Dropout(0.3)(dense7) 
dense9 = Dense(77)(dense8) 
dense10 = Dropout(0.2)(dense9) 
output1 = Dense(100, activation = 'softmax')(dense10)

model = Model(inputs = input1, outputs=output1)

model.summary()

# 3. 컴파일, 훈련
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq=0, write_graph = True, write_images = True)

dnnmodelpath = './model/{epoch:02d}-{loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath = dnnmodelpath, monitor = 'loss', 
                             save_best_only=True, mode = 'auto') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, callbacks = [early_stopping, tb_hist, checkpoint])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)


# 시각화

plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()