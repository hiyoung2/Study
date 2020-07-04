import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.applications import MobileNet

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam


# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape :", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape :", x_test.shape) # (10000, 32, 32, 3)
print("y_train.shape :", y_train.shape) # (50000, 1)
print("y_test.shape :", y_test.shape) # (10000, 1)


# 1-1 데이터 전처리
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 2 모델 구성
model = Sequential()

model.add(MobileNet(include_top = False, input_shape = (32, 32, 3)))
model.add(Flatten())
model.add(Dense(256, kernel_regularizer=l1(0.001)))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128, kernel_regularizer=l1(0.001)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32, verbose = 1, validation_split = 0.3)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

print("LOSS :", loss)
print("ACC :", acc)


# 5. VISUALIZATION, 시각화
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
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(acc, marker = '.', c = 'green', label = 'acc')
plt.plot(val_acc, marker = '.', c = 'purple', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

'''
LOSS : 0.9189712267875672
ACC : 0.7975999712944031

과적합 방지할 수 있는 요소들 추가
loss-val_loss, acc-val_acc 차이 줄어듦 
그래프가 비슷하게 진행
'''