# npy로 저장한 mnist 파일을 불러들이기 

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 저장해놓은 데이터들을 불러옴, 따라서 지금까지 데이터를 불러왔던 것처럼 위에 불러올 필요가 없이 
# load한 데이터들을 가지고 모델을 구성하면 된다
# 앞선 파일을 복제했기 때문에 위에 있었던 데이터들을 삭제 후 진행 중

x_train = np.load('./data/mnist_train_x.npy')
x_test = np.load('./data/mnist_test_x.npy')
y_train = np.load('./data/mnist_train_y.npy')
y_test = np.load('./data/mnist_test_y.npy')

print('x_train.shape : ', x_train.shape)  # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)    # (10000, 28, 28)
print('y_train.shape : ', y_train.shape)  # (60000,)
print('y_test.shape : ', y_test.shape)    # (10000,)

# 데이터 전처리 1. 원핫인코딩
# y data
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)

# 데이터 전처리 2. 정규화
# x data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape) # (60000, 28, 28, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(77, (2, 2), input_shape = (28, 28, 1)))     
model.add(Conv2D(111, (3, 3), activation = 'relu'))
model.add(Dropout(0.2))     

model.add(Conv2D(99, (3, 3), padding = 'same'))   
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))          

model.add(Conv2D(55, (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

# model.save('./model/model_test01.h5')

# 3. compile, 훈련
# early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto') 

# modelpath = './model/check--{epoch:02d}--{val_loss:.4f}.hdf5'
# checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
#                              save_best_only = True, save_weights_only = False, verbose = 1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=200, validation_split = 0.2,
                                   verbose = 1) 

# model.save('./model/model_test01.h5')


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
print('loss : ', loss)
print('acc : ' , acc)
# print('val_acc : ', val_acc)
# print('val_loss : ', val_loss)

# y_pred = model.predict(x_test)


# 시각화 
plt.figure(figsize = (10, 6)) 

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
plt.grid() 
plt.title('loss')      
plt.ylabel('loss')      
plt.xlabel('epoch')          
# plt.legend(['loss', 'val_loss']) 
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '*', c = 'purple', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '*', c = 'green', label = 'val_acc')
plt.grid() 
plt.title('acc')      
plt.ylabel('acc')      
plt.xlabel('epoch')          
plt.legend(['acc', 'val_acc']) 
plt.show()  

