#1. 데이터 준비 
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])    

# line에 커서 두고 ctrl+c , ctrl+v 하면 라인 전체 복붙 
# shift+del : line 전체삭제

#2. 모델구성                        (작업에 대한 주석 달아주면 좋음)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()                # model_young = Sequential 도 가능, Sequential의 model_young으로 하겠다는 의미
                                    # 모델을 하나 만드는데  sequential , 즉 순차적인 모델을 만들겠다

# model.add(Dense(5, input_dim = 1))  # 여기서 일단, activation 쓰지 않음.(이것은 default값 존재함을 의미/ 추후 설명) / parma :5 (bias 제외)
# model.add(Dense(3))                 # 1이 아니기 때문에 output이 아니라 hidden layer임을 알 수 있음 / param : 15
# # model.add(Dense(1000000))   # param : 3000000  (bias 제외) / 실행o
# # model.add(Dense(1000000))   # param : 1000000^2(bias 제외) / 실행 x / GPU에선 실행 될까? 안 될까?
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))   # 코딩 자체엔 문제 없음. 터질까 안 터질까? 해보면 됨
# model.add(Dense(9))
# model.add(Dense(4))
# model.add(Dense(13))
# model.add(Dense(7))
# model.add(Dense(11))
# model.add(Dense(15))
# model.add(Dense(9))
# model.add(Dense(18))
# model.add(Dense(3))
# model.add(Dense(1))         # param : 1000000 

# loss와 acc는 항상 상대적. loss 높고 acc 높을 수도 있고 loss 낮고 acc 낮을 수도 있음

model.add(Dense(10, input_dim = 1))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(21))
model.add(Dense(23))
model.add(Dense(24))
model.add(Dense(22))
model.add(Dense(20))
model.add(Dense(18))
model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련   / computer no. machine 훈련 시킴
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # metrics는 accuracy를 쓴다, 대괄호를 쓰는 건 문법
model.fit(x, y, epochs=30, batch_size=1)    # batch_size는 default 값(32) 있어서 생략해도 프로그램은 실행 된다

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)    # x, y를 넣어서 이 model을 평가하겠다, 평가 결과를 loss, acc에 받겠다. 
                                                  # loss, acc는 변수(variable). loss22, acc333 이런 식으로 쓸 수도 있지만!
                                                  # 통상적으로 이해하기 쉽게 써야하기 때문에 loss, acc 이렇게 써 주는 것이 좋다.
print("loss : ", loss)
print("acc : ", acc)

"""
model.add(Dense(10, input_dim = 1))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(29))
model.add(Dense(28))
model.add(Dense(27))
model.add(Dense(26))
model.add(Dense(25))
model.add(Dense(24))
model.add(Dense(23))
model.add(Dense(22))
model.add(Dense(21))
model.add(Dense(1))
epochs=100, batch_size=1
loss :  2.6557600563137385e-11
acc :  1.0


model.add(Dense(10, input_dim = 1))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(16))
model.add(Dense(17))
model.add(Dense(18))
model.add(Dense(19))
model.add(Dense(20))
model.add(Dense(22))
model.add(Dense(21))
model.add(Dense(1))
epochs = 200, batch_size = 1
loss :  1.9014123608940283e-12
acc :  1.0


model.add(Dense(3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(21))
model.add(Dense(20))
model.add(Dense(18))
model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epochs = 100, batch_size = 1

loss :  6.792788553866557e-13
acc :  1.0

"""