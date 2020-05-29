import numpy as np  
import matplotlib.pyplot as plt

from keras.datasets import cifar100 # column 100개 짜리 예제 ㅎㄷㄷ(현존하는 가장 큰 놈)
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Conv2D, Dense
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)  
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)  

plt.imshow(x_train[3])
plt.show()

# early_stopping, checkpoint, tensorboard 모두 쓰기
# cnn, dnn, lstm 파일 3개 추가 생성, 모두 dropout 적용
# 시각화도 epoch 100개 정도하자
# reshape 말고 차원 축소 배울 예정, 성능엔 변함 없고, 속도는 향상됨