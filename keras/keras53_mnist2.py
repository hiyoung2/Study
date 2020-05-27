# mnist

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print('x_train : ', x_train[0])
print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000,)
print('y_test.shape : ', y_test.shape)   # (10000, )

print(x_train[0].shape)
# print(y_train[0])
# plt.imshow(x_train[0], 'gray') 
# plt.imshow(x_train[0]
# plt.show()

# 분류모델, 0 ~ 9까지 하려면 10개로 분류해야함, 0인지 1인지, ... 9인지 판별해야함, 10가지의 경우의 수가 있다
# 현재 y는 1차원

# 데이터 전처리 1. 원핫인코딩
# y data
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2. 정규화
# x data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
# reshape으로 4차원 만들고(CNN 모델 집어 넣기 위해 cnn input shape : (batch_size 무시) 가로, 세로, 채널)
# 타입 바꾸고, 원래 타입은 int 형으로 0 ~ 255까지(255:완전 찐한 검정, 0:흰색) -> float 타입으로 변환
# minmax는 0부터 1까지, 실수이기 때문에 float 형으로 바꿔준다
# minmax 까지 적용(정규화, minmax scaler), 3가지를 한 번에 처리
# x_train = x_train / 255 # 최대가 255니까 255로 나누면 최댓값 1, 최솟값 0이 되니까 minmaxscaler 쓴 효과와 거의 동일(약간의 차이는 있다)
# 지금은 데이터 수를 아니까 minmaxscaler 대신 직접 255로 나눌 수 있음
# 셀 수 없이 많은 데이터면 언제 그거 다 세서 직접 나누겠냐, minmaxsclaer 적용시켜야함

print(x_train.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape = (28, 28, 1)))     
model.add(Conv2D(82, (3, 3), activation = 'relu'))                                 
model.add(Conv2D(80, (3, 3)))               
model.add(Conv2D(46, (2, 2), activation = 'relu'))
                                 

model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax')) # 최종 아웃풋 10, 10개 중에 하나 
                                             # softmax 통과하기 전에 데이터는 어떤 상태였나?
                                             # Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 
                                             # 출력 값들의 총합은 항상 1이 되는 특성을 가
                                             # 진 함수이다.
                                             # sigmoid는 0 아니면 1, 다중분류에 맞지 않다
                                             # softmax는 자리별로 가장 최적 값을 알려줌, 인덱스!
                                             
                                             # 원핫인코딩 원래 (6만, ) 이던 벡터 형태를
                                             # (6만, 10) 2차원으로 바꿔줌, 행 무시니까 마지막 아웃풋 디멘션이 10이 된다

model.summary()

# 3. compile, 훈련
# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=5, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=100, validation_split = 0.2) 

# 4. 예측, 평가

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)

print('loss : ', loss)
print('acc : ' , acc)

# x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

# acc 0.982로 올려라,,,,,



