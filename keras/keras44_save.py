# 0525 day11 - afternoon
# keras44_save.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(4,1)) )
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

model.summary()

# 모델 저장하기!!!
# model.save(".//model//save_keras44.h5") # h5: model save 확장명, .: 현재 경로
model.save("./model/save_keras44.h5")
# model.save(".\model\save_keras44.h5")
# 3가지 방법 모두 다 된다

print("저장 성공") # 경로 설정 안 해주니까 기본 작업 공간에 저장 됨, 지정된 곳에 하고 싶으면 경로 지정을 해 줘야 한다
