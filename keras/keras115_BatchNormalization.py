# 과적합 방지 : BatchNormalization
# wx + b 계산 과정에서 값을 0 ~ 1 사이로 수렴시키겠다
# BatchNomarlization 사용 시에는 별도로 Activation을 import 해야 한다
# Activation 이전에! BatchNormalization을 적용시켜야 한다
# 원래 목적 : BatchNormalization 을 통해 정규화한 값들을 활성화함수, activation으로 보내줌
# activation 이후에 쓰면 활성화 함수를 통과한 값에 적용된다 -> 목적과 다름
# 따라서 활성화함수 이전에 batchnormalization 사용!

# 활성화 함수의 역사
# 1) 계단 함수(무조건 0 아니면 1)
# 2) sigmoid(중간 손실을 방지)
# 3) relu(음수값 손실 문제)
# 4) leakyrelu(음수값이 계속 내려감)
# 5) elu(음수 상계에 대해 제한을 걸자)
# 6) selu(0과 1 사이로 밀어넣는다?)

# 규제나 제한에 있어서 발전했다고 더 좋은 것은 X (데이터에 맞춰서 사용해야 한다)
# 하이퍼파라미터 튜닝 자동화로 어느 정도 맞춰줄 수 있다

# 과적합 방지 기능 세 가지(L1, L2 / Dropout / BatchNormalization) 모두를 다 쓸 수는 있다
# But, 다 쓰면 성능이 무조건 좋아진다? Nope
# 어떤 경우에는 안 쓴 것이 나을 수도 있다,,,

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
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

model.add(Conv2D(32, kernel_size = 3, padding = 'same', input_shape = (32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size = 3, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(64, kernel_size = 3, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size = 3, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(128, kernel_size = 3, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size = 3, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
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
loss : 1.377245561027527
acc : 0.6761999726295471

별로 효과 X
'''