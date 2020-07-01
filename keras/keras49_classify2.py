import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )
print("y :", y) # y : [1 2 3 4 5 1 2 3 4 5]

y = y - 1
print("y :", y) # y : [0 1 2 3 4 0 1 2 3 4]

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print(y)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print("y.shape :", y.shape) # (10, 5)

# 10, 6이 나옴,, 10, 5가 나와야함 앞에 왜 0이 나오고 0을 제거해라

# 방법 1
# y = np.delete(y, 0, axis =1)
# print(y)
# print(y.shape)
# 방법 2
# y = y[:, 1:]
# print(y)
# print(y.shape)

# one-hot 인코딩 : 다중분류 시 필수로 해야하는 것

'''
# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation = 'relu'))
model.add(Dense(90))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20))                                  
model.add(Dense(5, activation = 'softmax'))   # (10, 6) 행 무시,  dimension = 6
                                              # 하나 넣었는데 6개 나옴 softmax 
                                              # 제일 큰 수 말고 다 0으로 나옴
                                              # 
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
print(np.argmax(y_pred, axis = 1)+1)
# numpy의 argmax는 디코딩에 해당


# a = model.predict([1,2,3,4,5])
# print(np.argmax(a, axis = 1)+1)

# 다중분류 모델에서 데이터 전처리에 꼭 필요한 one_hot_ecoding
# one-hot-encoding이란 단 하나의 값만 True, 나머지는 모두 False인 인코딩
# 즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다
# 예를 들면 [0,0,0,0,1]이다
# 5번째(zero-based index이니까 4)만 1이고 나머지는 0이다
# 데이터를 0 아니면 1 로 짝을 맞춰주기 위해서
# from keras.utils import np_utils
# y = np_utils.to_categorical(y)
# 를 써 주거나
# 
# from sklearn.preprocessing import OneHotEncoder # one-hot encoder 싸이킷런에 있음
# aaa = OneHotEncoder()
# aaa.fit(y)
# y = aaa.transform(y).toarray()
# 를 써줘야 하는데

# 차이점은 np_utils.to_categorical의 경우
# y data가 0부터 시작하지 않으면 슬라이싱 등을 통해 index를 조절해줘야 한다

# 반면 one-hot encoder는 0부터 시작하지 않더라도 알아서 데이터 shape에서 첫 번째 열에 0을 없애주고 
# 모델에 넣을 수 있는 형태로 만들어주는데, 이 때 중요한 것은 차원을 맞춰줘야 한다는 거다
# 예를 들면
# y = np.array([1,2,3,4,5,1,2,3,4,5])
# print(y.shape) # (10, ) 현재 1차원 vector 형태
# # one-hot encoder 는 2차원 형태로 넣어줘야 함
# # y = y.reshape(-1, 1) # -1? : 제일 끝, 
# == y = y.reshape(10, 1) # -1과 10 같음
# 2차원으로 변형!
'''