from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print("x.reshape : ", x.shape)

print(x)

# 2. 모델 구성
model = Sequential()

model.add(SimpleRNN(10, activation = 'relu', input_shape = (3, 1)))
# == model.add(SimpleRNN(10), activation = 'relu', input+length = 3, input_dim = 1))
model.add(Dense(20))
model.add(Dense(9))
model.add(Dense(1))

model.summary()

# 3. compile, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 300, batch_size = 1, verbose = 0)

# 4. 실행
x_predict = array([5, 6, 7])
x_predict = x_predict.reshape(1,3,1)

# 5. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print("결괏값 : ", y_predict)