# sigmoid를 모든 레이어마다 쓴다면?

# 1. data
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. modeling
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_shape = (1, )))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# h1(x) = w1*x + w2*x + ... + b 가 activation에 들어가서 계산
# 각 layer 에서 계산된 결괏값이 activation을 통해 다음 layer로 전달 된다
# activation's default = linear

# sigmoid를 가장 위의 layer에 사용하지 않는 이유?
# sigmoid는 0과 1 사이로 수렴시키는 것인데 각 레이어마다 0.5(?)를 곱하면 0에 수렴하는 문제가 생긴다(???)
# 'relu'는 식 자체가 무조건 0으로 수렴하는 것을 막아준다'


# 3. compile, training
model.compile(loss = ['binary_crossentropy'], optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# 4. evaluation, prediction
loss =  model.evaluate(x_train, y_train)
print("LOSS :", loss)

x1_pred = np.array([11, 12, 13, 14])
y_pred = model.predict(x1_pred)
print("예측값 :", y_pred)

'''
예측값 : [[1.]
 [1.]
 [1.]
 [1.]]
'''