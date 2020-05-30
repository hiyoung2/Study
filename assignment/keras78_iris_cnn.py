import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.decomposition import PCA

# 1. 데이터 준비
iris = load_iris()
x = iris['data']
y = iris['target']

print(x)
print(y)

print('x.shape : ', x.shape) # (150, 4)
print('y.shape ; ', y.shape) # (150,)

# 1.1 데이터 전처리
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (150, 3)

# 1.2 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print('x_scaled : ', x.shape) # (150, 4)

# column이 4개밖에 안 되므로 안 하는 게 좋을 것 같지만 연습삼아 시도
# pca = PCA(n_components = 2)
# pca.fit(x)
# x_pca = pca.transform(x)
# print(x_pca.shape) # (150, 2)


# 1.3 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

# 1.4 데이터 shape 맞추기
x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)

# 2. 모델 구성
model = Sequential()

model.add(Conv2D(11, (2, 2), input_shape = (2, 2, 1)))
model.add(Conv2D(22, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(44, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(66, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(99, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(111, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(88, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(55, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(33, (2, 2), padding = 'same', activation = 'relu'))
model.add(Conv2D(11, (2, 2), padding = 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 200, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, acc  = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
print(y_pred)
print(np.argmax(y_pred, axis = 1))


'''
loss :  0.15457082632152203
acc :  1.0
'''