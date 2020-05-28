# cifar kikiki

from keras.datasets import cifar10 # 여기에 , mnist 써도 상관은 없다, 명확하게 하기 위해 하는 것만 넣어주고 귀찮으면 다 써 줘도 된다
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Conv2D, Dense
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np  

# 총집합,,,

# 동물 10종류, 컬러 그림 모음집
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape) # (50000, 32, 32, 3)
print("x_test.shape : ", x_test.shape)   # (10000, 32, 32, 3)
print("y_train.shape : ", y_train.shape) # (50000, 1)
print("y_test.shape : ", y_test.shape)   # (10000, 1)

plt.imshow(x_train[3])
plt.show()