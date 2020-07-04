# 과적합 방지 : Dropout

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# 1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape :", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape :", x_test.shape) # (10000, 32, 32, 3)
print("y_train.shape :", y_train.shape) # (50000, 1)
print("y_test.shape :", y_test.shape) # (10000, 1)

# 1-1 DATA PREPROCESSING
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print("y_train.shape :", y_train.shape) # (50000, 10))
# print("y_test.shape :", y_test.shape) # (10000, 10)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 2. MODELING
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3. COMPILE, TRAINING
# sparse_categorical_crossentropy 를 loss 함수로 쓰려면 
# y data -> one-hot encoding 실행하면 안 됨 (one-hot encoidng 하고 spares~ 쓰면 SHAPE ERROR 발생)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(1e-4), metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.3)

# 4. Evaluation, Prediction
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

print("loss :", loss)
print("acc :", acc)

# 5. VISUALIZATION
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(loss, marker = '.', c = 'red', label = 'loss')
plt.plot(val_loss, marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(acc, marker = '.', c = 'green', label = 'acc')
plt.plot(val_acc, marker = '.', c = 'purple', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()

plt.show()



'''
loss : 0.7302180895328522
acc : 0.758400022983551

그래프가 거의 일치하는 모습
과적합 방지가 된 듯
'''