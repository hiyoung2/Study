from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input

# a08 copy
# cnn으로 autoencoder 구성하시오

def autoencoder() :

    input_img = Input(shape=(28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

def autoencoder_sq() :
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters = 1, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same'))

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
# model.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
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

plt.tight_layout() # figure 상에서 배치 되어 있는 것들의 공백을 적당하게 잘 배치해주는 method
plt.show()


'''
sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 16)        160
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 8)         1160
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 8)           0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 8)           0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 8)           584
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 8, 8, 8)           0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 8)           584
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 16, 16, 8)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 16)        1168
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 28, 28, 16)        0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 28, 28, 1)         145
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
_________________________________________________________________
'''

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 16)        160
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 8)         1160
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 8)           0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 8)           0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 8)           584
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 8, 8, 8)           0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 8)           584
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 16, 16, 8)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 16)        1168
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 28, 28, 16)        0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 28, 28, 1)         145
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
_________________________________________________________________
'''