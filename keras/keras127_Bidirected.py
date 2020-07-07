# 2020.07.07
# 126 파일 copy

# bidirectional :  양방향!
# 과거------>현재 순
# 과거<------현재 순
# 시계열 작업이 2번 가능
# 연산의 양이 2배가 된다

from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 20000) # test_split = 0.2 
print("x_train.shape :", x_train.shape)  # (25000,)
print("x_test.shape :", x_test.shape)    # (25000,)
print("y_train.shape :", y_train.shape)  # (25000,)
print("y_test.shape :", y_test.shape)    #  (25000,) 

print(x_train[0])
print(y_train[0]) # 1
print(y_train[1]) # 0

print("x_train[0]의 크기 :", len(x_train[0]))  # 218
print("x_train[-1]의 크기 :", len(x_train[-1])) # 153
# y의 카테고리 개수 출력
category = np.max(y_train) + 1 
print("카테고리 :", category) # 2  

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train) 
print(y_bunpo) # [0 1]

# y_train 데이터 값들이 어떻게 분포되어 있는지 확인
y_train_pd = pd.DataFrame(y_train)     # y_train을 padas 형식에 집어넣어준다
bbb = y_train_pd.groupby(0)[0].count() # 주간과제 : pandas의 groupby() keyword 공부!! 사용법 숙지
print(bbb)

'''
0    12500
1    12500
Name: 0, dtype: int64
'''
print("bbb.shape :", bbb.shape) # bbb.shape : (2,)

# 인덱스를 단어로 바꿔주는 함수
word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

print("빈도수 상위 1번 단어 : {}".format(index_to_word[1])) # 빈도수 상위 1번 단어 : the
print("빈도수 상위 2000번 단어 : {}".format(index_to_word[2000])) # 빈도수 상위 2000번 단어 : behavior


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 150, padding= 'pre')
x_test = pad_sequences(x_test, maxlen = 150, padding= 'pre')

# 0과 1로 이루어져 있으므로
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print("x_train.shape :",x_train.shape)  # (8982, 100)
print("x_test.shape :", x_test.shape)   # (2246, 100)
print("y_train.shaep :", y_train.shape) # (8982, 46)
print("y_test.shape :", y_test.shape)   # (2246, 46)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.layers import Conv1D, MaxPooling1D, Bidirectional


model = Sequential()
# model.add(Embedding(2000, 128, input_length = 111))
model.add(Embedding(2000, 128))
model.add(Conv1D(10, 5, padding = 'valid', activation ='relu', strides = 1))
model.add(MaxPooling1D(pool_size = 4))

model.add(Bidirectional(LSTM(10))) # 1680
# model.add(LSTM(10)) # 840 # 파라미터수가 2배 차이 난다

# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit(x_train, y_train, batch_size = 128, epochs = 16, validation_split = 0.2)

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
