from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose


def autoencoder() :
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size = (15, 15), padding = 'valid', input_shape = (28, 28, 1), activation = 'relu'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(Conv2D(filters = 32, kernel_size = (4, 4), padding = 'valid', activation = 'relu'))
    # # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2D(filters = 64, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 128, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 256, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'valid'))
    model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'))

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

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

print("x_train.reshape :", x_train.shape)
print("x_test.reshape :", x_test.shape)

# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
model = autoencoder()
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
    plt.subplots(3, 5, figsize = (15, 5))

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


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        4640
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 22, 22, 64)        18496
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 20, 20, 128)       73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 18, 18, 256)       295168
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 64)        147520
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 14, 14, 32)        18464
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 28, 28, 32)        0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 1)         289
=================================================================

bad
'''


'''
def autoencoder() :
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size = (15, 15), padding = 'valid', input_shape = (28, 28, 1), activation = 'relu'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(Conv2D(filters = 32, kernel_size = (4, 4), padding = 'valid', activation = 'relu'))
    # # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2D(filters = 64, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 128, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 256, kernel_size = (4, 4), padding = 'valid'))
    # model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'valid'))
    model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'))

    return model

Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 14, 14, 8)         1808
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 28, 28, 8)         0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 1)         73
=================================================================
Total params: 1,881
Trainable params: 1,881
Non-trainable params: 0
_________________________________________________________________

dnn보다 별로 -> 다시 수정 필요

'''