# a 2345는 AUTO ENCODER가 아니다, 개념 설명을 위한 과정이었음

# keras 2.4.0 version이 최근 release 되었다
# 더이상 multi backend를 지원하지 않는다
# 2.4.0 version은 기존 구현 코드를 모두 삭제하고 대신 tensorflow.keras로 redirection 한다

# 일단 나는 현재 2.3.1 version 사용중

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# autoencoder 함수 정의
# 함수 정의 : def, return 

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

print("x_train.reshape :", x_train.shape)
print("x_test.reshape :", x_test.shape)


# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
model = autoencoder(hidden_layer_size=154)
# model = autoencoder (32)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# loss: 0.0102 - acc: 0.0110(32)
# loss: 0.0014 - acc: 0.0163 (154)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
# loss: 0.0938 - acc: 0.8141 (32)
# loss: 0.0657 - acc: 0.8155 (154)


# mse, binary_crossentropy 둘 다 적용 시켜 봐야 한다

model.fit(x_train, x_train, epochs=10, batch_size = 128) 

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()