# 112 번 파일 : 과적합
# 과적합 방지? L1, L2규제

# 과적합을 해결하는 방법 3가지
# 1) 훈련 데이터를 늘린다
# 2) 피쳐수를 늘린다
# 3) 제약을 가한다

'''
과적합의 이유?
연산한 값이 튀는 경우가 생긴다 = Gradient Exploding or Gradient Vanishing
가중치 값, 그 자체가 너무 커지게 되면 예를 들어, w = 100만이라면 relu를 만나면 그대로 위로 간다(그래프를 생각)
그리고 그 다음 레이어의 노드에 *, 곱해진다, 그럼 그 노드 값과 또 엄청난 상승이 이뤄진다
그러면 가중치 하나만으로 컨트롤이 되지 않는다
-> 활성화 함수가 어느 정도 제어를 해준다고 하지만, 그것만으로는 부족

초기에 최초로 나온 것이 sigmoid, but 완전한 해결 X
(가중치 연산이 이루어지는 과정에서 엄청난 큰 값이 자꾸 생성되는 걸 막기가 힘듦)
이것을 제어하기 위해 Regularizer가 생겼다
'''

'''
layer에 kernel_regularizer

L1 규제 : 가중치의 절댓값 합
regularizer.l1(1=0.01)
희소 특성에 의존하는 모델에서 관련성이 없거나 매우 낮은 특성의 가중치를 정확히 0으로 유도하여 모델에서 해당 특성을 배제
L2와 반대되는 개념

L2 규제 : 가중치의 제곱 합
regularizer.l2(1 = 0.01)
가중치 행렬의 모든 원소를 제곱하고 0.01을 곱하여 네트워크의 전체 손실에 더해진다는 의미
이 규제(penalty)는 훈련할 때만 추가된다
높은 긍정 값 또는 낮은 부정 값을 갖는 이상점 가중치를 0에 가까이 유도
L1와 반대되는 개념

loss = L1 * reduce_sum(abs(x))
loss = L2 * reduce_sum(square(x))
'''

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

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
# kl2 = l1(0.001)
# kl2 = l2(0.001)

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(256, activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. COMPILE , TRAINING
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(1e-4), metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.3)


# 4. Evaluation, Prediction
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

print("LOSS :", loss)
print("ACC :", acc)


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
LOSS : 1.4514734603881836
ACC : 0.5839999914169312

# 112번 파일의 그래프보다 과적합 부분이 나아짐
'''