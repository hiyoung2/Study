from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose

# a08 copy
# cnn으로 autoencoder 구성, padding = 'valid'를 사용

def autoencoder_sq() :
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid', input_shape = (28, 28, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'valid'))

    # model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'valid', activation = 'sigmoid'))

    return model
    

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train / 255
x_test = x_test / 255

print("x_train.reshape :", x_train.shape)
print("x_test.reshape :", x_test.shape)


# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
# model = autoencoder()
model = autoencoder_sq()
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model.fit(x_train, x_train, epochs=50, batch_size = 200, validation_split = 0.2) 

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

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 16)        160
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 16)        0
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 26, 26, 16)        0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 1)         145
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 8)         80
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 8)         0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 26, 26, 8)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 1)         73
=================================================================
'''