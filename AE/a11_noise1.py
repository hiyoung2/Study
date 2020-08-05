from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784, ), activation  = 'relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

x_train = x_train / 255
x_test = x_test / 255

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)

# randon.normal(평균, 표준편차, shape)
# 정규분포로부터 무작위 표본 추출
# x_train 픽셀들에게 값을 더해줌으로써 원래 깨끗한 데이터에 noise를 발생시킨다
# 평균 0, 표준편차 0.5인 정규분포를 적용하면 많은 픽셀들이 0에 거의 수렴해서
# 지나치게 큰 양수나 음수가 나오진 않겠지만
# 어찌 됐든 x_train은 0 ~ 1 사이인데 거기에 숫자들을 더하면
# 음수도 발생할 것이고 1을 넘어가는 값들도 생김

# 이전 단계에서 255로 나눠줘서 0 ~ 1 사이로 맞춰주었는데(MinmaxScaler 효과)
# random.normal을 써 버리면 우리가 스케일링 해 놓은 범위를 넘어가는 값들이 발생한다
# 의도하지 않은 문제가 발생할 수 있다
# 그래서 다시 0 ~ 1 사이로 맞춰줘야 한다

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

print("x_train.reshape :", x_train.shape)
print("x_test.reshape :", x_test.shape)

# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
model = autoencoder(hidden_layer_size = 16)
# model = autoencoder (32)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc']) # loss = mse 
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

model.fit(x_train_noised, x_train, epochs=10, batch_size = 128) 

output = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random

# 이미지 3종류를 살펴보자
# 원본이미지(NOISE x)(test)(1), Noise이미지(test_noise)(2), Output이미지(output)(3)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# (1) 원본 이미지를 맨 위에 그린다(x_test)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("ORIGIN", size = 30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# (2) 원본 + noise(x_test_noised)
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size = 30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])    

# (3) 오토 인코더가 출력한 이미지를 마지막에 그린다(output)
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size = 30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


