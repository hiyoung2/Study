from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000, test_split = 0.2) # 가장 많이 쓰는 단어 1000개를 가져오겠다, 20% test data로 쓰겠다

print("x_train.shape :", x_train.shape) # (8982,) # print(x_train.shape, x_test.shape) 이렇게 봐도 됨
print("x_test.shape :", x_test.shape)   # (2246,)
print("y_train.shape :", y_train.shape) # (8982,)
print("y_test.shape :", y_test.shape)   # (2246,)

print(x_train[0])
print(y_train[0]) # 3 ex) 날씨 (정치, 스포츠 등등)

# print(x_train[0].shape) # list라서 shape 먹히지 않음
print(len(x_train[0]))  # 87
print(len(x_train[-1])) # 105
# y의 카테고리 개수 출력
category = np.max(y_train) + 1  # index는 0부터 시작하므로 1을 더해서 표기해준다
print("카테고리 :", category)   # 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train) # y_train data에서 unique한 것들만 보여준다(즉 하나씩이니까 데이터가 어떻게, 어디까지 분포되어있는지 확인 가능)
print(y_bunpo)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# y_train 데이터 값들이 어떻게 분포되어 있는지 확인
y_train_pd = pd.DataFrame(y_train)     # y_train을 padas 형식에 집어넣어준다
bbb = y_train_pd.groupby(0)[0].count() # 주간과제 : pandas의 groupby() keyword 공부!! 사용법 숙지
print(bbb)
print("bbb.shape :", bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 150, padding= 'pre')
x_test = pad_sequences(x_test, maxlen = 150, padding= 'pre')
# truncating='pre' / truncating : 자르는 것! 근데 명시 안 해도 pre로 dafault 적용됨
# maxlen도 명시 안 해줘도 돌아감 : 아마 가장 큰 값에 자동으로 맞출 듯

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train.shape :",x_train.shape)  # (8982, 100)
print("x_test.shape :", x_test.shape)   # (2246, 100)
print("y_train.shaep :", y_train.shape) # (8982, 46)
print("y_test.shape :", y_test.shape)   # (2246, 46)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D

model = Sequential()
# model.add(Embedding(1000, 128, input_length = 100))
model.add(Embedding(10000, 256))
model.add(Conv1D(512, 5, padding = 'valid', activation ='relu', strides = 1))
model.add(MaxPooling1D(pool_size = 4))
model.add(LSTM(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(46, activation = 'softmax'))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit(x_train, y_train, batch_size = 128, epochs = 32, validation_split = 0.2)

# 4. 평가
acc = model.evaluate(x_test, y_test)[1]
print("ACC :", acc)

# 5. 시각화
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TrainSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

'''
model = Sequential()
model.add(Embedding(1000, 100, input_length = 100))
model.add(LSTM(5))
model.add(Dense(46, activation = 'softmax'))

ACC : 0.5271593928337097
'''
'''
model.add(Embedding(1000, 128))
model.add(LSTM(64))
model.add(Dense(46, activation = 'softmax'))
ACC : 0.5841495990753174
'''
'''
model = Sequential()
model.add(Embedding(1000, 128, input_length = 100))
model.add(LSTM(64))
model.add(Dense(46, activation = 'softmax'))
ACC : 0.6424754858016968
'''

'''
model.add(Embedding(1000, 256))
model.add(LSTM(128))
model.add(Dense(64))
model.add(Dense(46, activation = 'softmax'))
ACC : 0.6727515459060669
'''
'''
model.add(Embedding(10000, 256))
model.add(LSTM(128))
model.add(Dense(64))
model.add(Dense(46, activation = 'softmax'))

ACC : 0.6304541230201721
'''

'''
num_words = 10000
model.add(Embedding(10000, 128))
model.add(Conv1D(512, 5, padding = 'valid', activation ='relu', strides = 1))
model.add(MaxPooling1D(pool_size = 4))
model.add(LSTM(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(46, activation = 'softmax'))

batch = 128, epoch = 32
ACC : 0.6803205609321594
'''

'''

'''