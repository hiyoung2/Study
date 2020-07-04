# 현재 loss를 줄이는 작업
# 딥러닝에서 loss는 model.compile에 존재

# 1. 데이터

import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

# 3. 컴파일, 훈련
# 최적화 함수를 import

# RMSprop : 학습률을 제외한 모든 인자의 기본값을 사용하는 것 권장
# SGD : 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 
# Adadelta : Adagrad 확장버전
# Adagrad : 모델 파라미터별 학습률을 사용함
# Nadam : 네스테로프(누적된 과거 Gradient가 지향하고 있는 어떤 방향을 현재 Gradient에 보정하려는 방식), Adam Optimizer

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

# opti = Adam(lr = 0.001) 
# loss : [0.9964981079101562, 0.9964981079101562] [[2.1646469]]

# opti = RMSprop(lr = 0.001)
# loss : [0.0011731685372069478, 0.0011731685372069478] [[3.4820774]]

# opti = SGD(lr = 0.001)
# loss : [0.015366243198513985, 0.015366243198513985] [[3.4418526]]

# opti = Adadelta(lr = 0.001)
# loss : [4.835816383361816, 4.835816383361816] [[0.68938756]]

# opti = Adagrad(lr = 0.001)
# loss : [5.645695209503174, 5.645695209503174] [[0.44620728]]

opti = Nadam(lr = 0.001)
# loss : [0.7393741607666016, 0.7393741607666016] [[2.348458]]

model.compile(loss = 'mse', optimizer = opti, metrics = ['mse'])
model.fit(x, y, epochs = 100)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)

pred = model.predict([3.5])
print(pred)