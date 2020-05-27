# 과제 2) 결괏값 깔끔하게 만들기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print(y)
print("y_shape : ", y.shape) 

# 10, 6이 나옴,, 10, 5가 나와야함 앞에 왜 0이 나오고 0을 제거해라

# 방법 1
# y = np.delete(y, 0, axis = 1) # 리스트의 각 행마다 index 0 [0]의 자리를 삭제
# print(y)
# print(y.shape)
# 방법 2
y = y[:, 1:]
print(y)
print(y.shape) # 슬라이싱으로 output은 5로 맞춰줌

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation = 'relu'))
model.add(Dense(90))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20))                                  
model.add(Dense(5, activation = 'softmax'))  
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=500, batch_size=1) 

# 4. 평가, 예측
loss, acc = model.evaluate(x, y)

x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)

print("loss : ", loss)
print("acc", acc)
print(y_pred)
# print(np.argmax(y_pred, axis = 1)+1)


# x_pred에 상응하는 y_pred가 5개씩 출력됨
# 그 중에서 1에 가까운 가장 큰 수가 y_pred에 해당하는 값인데
# 리스트 안에 값들이 들어있으니까
# 그 중 큰 수 하나를 뽑아서 결괏값으로 보여줘야한다
# argmax라는 함수를 쓰면 리스트 중 최댓값 하나를 뽑아서 출력해주는데
# 이 때 인덱스 때문에 0, 1, 2, 3, 4 로 나타나므로
# argmax 결과에 1을 더해주면 된다
# 여기서 axis = 1이란?
# axis는 좌표축과 같다 
# NumPy 함수의 인수로 axis를 설정하는 경우가 많다
# 예를 들어
# [[1 2 3] 
#  [4 5 6]] 이러한 2차원 배열이 있다면
# 열마다 처리하는 축 즉, 1-4, 2-5, 3-6 이렇게 묶어 처리하는 것은 axis=0이고
# 1-2-3, 4-5-6 이렇게 행마다 처리하는 축은 axis = 1이다
# 지금 여기에서는 각 행에서 최댓값을 뽑아주는 것이므로 axis = 1을 썼다

# a = model.predict([1,2,3,4,5])
# print(np.argmax(a, axis = 1)+1)
