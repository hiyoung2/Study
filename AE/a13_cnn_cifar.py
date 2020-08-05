# cifar10으로 autoencoder CNN 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D

# autoencoder 함수 정의
# 함수 정의 : def, return 

def autoencoder() :
    # 1
    # model = Sequential()
    # model.add(Conv2D(filters = 256, kernel_size = (17, 17), padding = 'valid', input_shape = (32, 32, 3), activation = 'relu'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'))

    # 2
    # model = Sequential()
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid', input_shape = (32, 32, 3), activation = 'relu'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
    # model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'))

    # 3
    # model = Sequential()
    # model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'valid', input_shape = (32, 32, 3), activation = 'relu'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'valid', activation = 'sigmoid'))

    # 4
    # model = Sequential()
    # model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'valid', input_shape = (32, 32, 3), activation = 'relu'))
    # model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
    # model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'valid'))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
    # model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(UpSampling2D(size = (2, 2)))
    # model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'))

    return model

from tensorflow.keras.datasets import cifar10

train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print("x_train.shape :", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape :", x_test.shape) # (10000, 32, 32, 3)


# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

x_train = x_train / 255.0
x_test = x_test / 255.0

# print("x_train.reshape :", x_train.shape)
# print("x_test.reshape :", x_test.shape)


# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
model = autoencoder()
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 

# mse, binary_crossentropy 둘 다 적용 시켜 봐야 한다

model.fit(x_train, x_train, epochs=20, batch_size = 256) 

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(32, 32, 3), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size = 10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(32, 32, 3), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size = 10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


'''
# 1
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 16, 16, 128)       111104
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 32, 32, 128)       0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 32, 32, 3)         3459
=================================================================
Total params: 114,563
Trainable params: 114,563
Non-trainable params: 0
_________________________________________________________________
엉망
'''

'''
#2
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 16)        448
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 16)        2320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 16)        2320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 16)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 32)          4640
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 8, 8, 32)          0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 8, 8, 32)          0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 32)          0
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 8, 8, 32)          0
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 16, 16, 32)        0
_________________________________________________________________
up_sampling2d_4 (UpSampling2 (None, 32, 32, 32)        0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 32, 32, 3)         867
=================================================================
Total params: 10,595
Trainable params: 10,595
Non-trainable params: 0
_________________________________________________________________
'''