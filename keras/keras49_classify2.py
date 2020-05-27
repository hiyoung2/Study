import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )
print(y.shape)



# print(y)
# print("y_shape : ", y.shape) 

y = y - 1

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print(y)
print(y.shape)

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


# a = model.predict([1,2,3,4,5])
# print(np.argmax(a, axis = 1)+1)
y