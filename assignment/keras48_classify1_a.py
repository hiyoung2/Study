# 과제 1) 결괏값 깔끔하게 만들기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )

# 2. 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 1)) # relu 평타 85점
model.add(Dense(36, activation='relu'))
# model.add(Dense(40, activation='relu'))  
# model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(18, activation='relu'))  
model.add(Dense(1, activation = 'sigmoid'))           
model.summary()


# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=350, batch_size=1) 

# rmsprop


# 4. 평가, 예측
loss, acc = model.evaluate(x, y) # loss와 metrics에 집어 넣은 값으로 평가

x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)

print("loss : ", loss)
print("acc", acc)
print(np.around((y_pred)))
# print(y_pred)

'''
acc 0.699999988079071
[[1.]
 [0.]
 [1.]]
'''



