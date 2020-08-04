# 지도학습(supervised learning)
# y 값, target 값이 존재하는 학습

# 비지도학습(unsupervised learning)
# target 값이 없음
# 배운 적이 있다? -> PCA
# PCA 할 때 x만 건드렸음

# x -> x
# 입력값, 출력값이 결국 같은 건데, 왜 하는 것인가?
# 특성, feature 추출이 이뤄졌다는 뜻, autoencoder 과정에서!
# 쓸 만한 feature들을 고르게 됨-> 압축이 일어난다

# # 예를 들어
# A 라는 사람의 사진을 input으로 넣어서 A 사진을 그대로 출력하고자 하는데
# 그 과정에서 잉크를 막 뿌렸다고 가정.
# auto encoder의 기대 결과는 잉크가 없는 깨끗한 A 사진.
# A 사진 == 데이터의 중요한 feature
# 여기서 잉크 == 잡음, 즉 노이즈가 된다.
# auto encoder 과정에서 노이즈가 제거된다!

# 특성이기 때문에 깔끔하게, 깨끗하게 나올 수는 없다
# 흐릿하게 나온다고 보면 된다
# 깔끔하게 나오게 하기 위해 뭔가 할 수 있다?
# -> GAN! auto encoder의 한계 극복 가능!

# 특성만 압축해서 작게 만들고 다시 원상태로 만든다
# 작게 만드는 것 : 인코더
# 다시 원래 상태로 만드는 것 : 디코더
# -> 이 과정이 "auto encoder"


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

# 마지막 활성화 함수 sigmoid를 사용하는 이유?
# 내가 그냥 쉽게 받아들인 것은
# 입력 - 출력 간의 두 개에 대한 data feature의 비교이기 때문에, 입력 된 데이터가 그대로 출력이 되었나, 안 되었나를 따지니까
# 쉽게 생각해서 이진 분류, 활성화함수는 sigmoid 사용
# 이렇게 생각하면 안되나,,,

# 일단, 설명에 따르면 feature 들을 minmaxscaler의 같은 효과를 가지도록 255로 나눠줬다
# 1 ~ 255 의 값을 가진 feature들에 255로 나눠줬기 때문에 0 ~ 1 사이가 된다
# 그에 맞춰 마지막 활성화 함수 sigmoid ( 0 ~ 1 수렴)를 사용한다


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

