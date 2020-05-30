import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 1. 데이터 준비
iris = load_iris()
x = iris['data']
y = iris['target']

print(type(iris)) # <class 'sklearn.utils.Bunch'>

# x = iris.data, y = iris.target으로 쓰니까 iris 부분에 빨간 줄이 생기면서 무언가를 경고
# 실행은 되니까 문제는 없는 건데,,,
# 해당 부분에 alt+F8 키로 뭐가 문제인지 보니까 Instance of 'tuple' has no 'target' member 이란 메세지가 떴다
# 구글링 해서 찾아본 결과 , x = iris['data], y = ['target'] 이라고 바꿔주니까 빨간 줄이 사라졌다!


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

# 1.3 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

# 1.3 데이터 shape 맞추기
# DNN 모델이라 따로 맞춰줄 필요 없음

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

# 2. 모델 구성
model = Sequential()

model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, acc  = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
print(y_pred)
print(np.argmax(y_pred, axis = 1))

'''
loss :  0.019614876400736374
acc :  1.0
'''