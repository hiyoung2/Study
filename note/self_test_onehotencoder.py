import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )
print(y)

# (10, ) # one-hot encoder 는 2차원 형태로 넣어줘야 함
y = y.reshape(-1, 1) # -1? : 제일 끝, 
# y = y.reshape(10, 1) # -1과 10 같음
# # 2차원으로 변형!

from sklearn.preprocessing import OneHotEncoder # one-hot encoder 싸이킷런에 있음
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()

print(y)
print(y.shape)
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
