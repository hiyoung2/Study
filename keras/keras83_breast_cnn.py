import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.decomposition import PCA

# 1. 데이터 준비
bc = load_breast_cancer()
x = bc['data']
y = bc['target']

print(x)
print(y)

print('x.shape : ', x.shape)  # (569, 30)
print('y.shape : ', y.shape)  # (569, )

# 1.1 데이터 전처리(One Hot Encoder)
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (569, 2)

# 1.2 데이터 전처리(Scaler)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# 1.3 데이터 전처리(PCA)
pca = PCA(n_components= 25)
pca.fit(x)
x_pca = pca.transform(x)

print('x_pca.shape : ', x_pca.shape) # x_pca.shape :  (569, 25)

# 1.4 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, train_size = 0.8
)

print('x_train.shape : ', x_train.shape) # x_train.shape :  (455, 25)
print('x_test.shape : ' , x_test.shape)  # x_test.shape :  (114, 25)

# 1.5 데이터 모양 맞추기

x_train = x_train.reshape(x_train.shape[0], 5, 5, 1)
x_test = x_test.reshape(x_test.shape[0], 5, 5, 1)

print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)


# 2. 모델 구성
model = Sequential()

model.add(Conv2D(55, (2, 2), input_shape = (5, 5, 1)))
model.add(Conv2D(77, (2, 2), padding= 'same', activation = 'relu'))
model.add(Conv2D(99, (2, 2), padding= 'same', activation = 'relu'))
model.add(Conv2D(111, (2, 2), padding= 'same', activation = 'relu'))
model.add(Conv2D(88, (2, 2), padding= 'same', activation = 'relu'))
model.add(Conv2D(44, (2, 2), padding= 'same', activation = 'relu'))
model.add(Conv2D(22, (2, 2), padding= 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y, axis = 1))

print('loss : ', loss)
print('acc : ', acc)

'''
loss :  0.21596390517005484
acc :  0.9736841917037964
'''