#  copy 56, mnist auto encoder 적용

import numpy as np


from tensorflow.keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# print('x_train : ', x_train[0])
# print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )

# print(x_train[0].shape)
# print(y_train[0])
# plt.imshow(x_train[0], 'gray') 
# plt.imshow(x_train[0]
# plt.show()


# # 데이터 전처리 - 정규화
# x data
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

# print(x_train.shape)


# # 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape = (784, ))
encoded = Dense(64, activation = 'relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)

autoencoder = Model(inputs = input_img, outputs = decoded)

autoencoder.summary()


# input_img = Input(shape = (784, ))
# encoded = Dense(32, activation = 'relu')(input_img)
# decoded = Dense(784, activation = 'sigmoid')(encoded)
# autoencoder = Model(inputs = input_img, outputs = decoded)
# autoencoder.summary()

# 위 모델 간단히 설명하면
# x : 6만, 784
# x^ : 6만, 784
# 중간에 32로 압축이 됨
# 1도 가능은 함


# 3. compile, 훈련
# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto')

autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split = 0.2) 

# autoencoder에서는 fit 과정에서 x_train과 x_train이 들어간다!


#4. 예측
# 비교하기 위한 예측 값 만들기
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20, 4))

for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()

