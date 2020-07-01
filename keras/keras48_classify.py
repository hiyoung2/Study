# 분류, classifier, classification

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# import 하는 건 상단에 적어주는 게 좋다

# 1. 데이터 준비
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

print("x.shpae : ", x.shape) # (10, )
print("y.shape : ", y.shape) # (10, )

# input_dim = 1 / scalar : 10, vector : 1

# 지금 데이터 선으로 그릴 수 있는가? nope
# x=1이면 y=1, x=2이면 y=0, ... 홀짝 홀짝 느낌
# 결과치가 두 가지 중에 하나! -> 분류모델
# 예 , 아니오 / 맞다, 틀리다 등등
# 살 지 안 살지 파악해주세요, 독성이 있는지 없는지 파악해주세요
# 결괏값이 두 가지로 한정, 어떤 숫자를 넣어도 결과는 1이나 0 한정되어있어야 한다 -> '이진분류 binary classification' 라고 한다

# 찾아야 할 것
# 우리는 activation을 디폴트로 썼음(디폴트 : relu??) 지금은 sigmoid를 써야 함
# 이진분류에서는 loss에 뭐가 들어갈까?
# optimizer 로 adam을 썼는데 이것도 이진분류에 맞는 게 있을 것


# 2. 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 1, activation = 'sigmoid')) # relu 평타 85점
model.add(Dense(36, activation = 'relu'))
model.add(Dense(60))
model.add(Dense(18))
model.add(Dense(20))                                    
model.add(Dense(1, activation = 'sigmoid'))             # activation 함수 전지전능, 모든 레이어에 강림하신다,,,
                                                        # activation 넣지 않아도 default 있음(linear)
                                                        # 레이어마다 y=wx+b가 계산된 값이 activation 함수를 거쳐서 다음 레이어에 전달을 해 준다
                                                        # 레이어마다 activation 함수를 명시하지 않아도 디폴트가 있어서 거쳐간다
                                                        # 마지막 output 최종값 * sigmoid : 0 아니면 1의 결과가 나온다(sigmoid의 기능)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=350, batch_size=1) 

# rmsprop
# loss='binary_crossentropy' 이진분류에서 loss는 이거 하나밖에 없음!!!!!!그냥 외우기

# 4. 평가, 예측
loss, acc = model.evaluate(x, y) # loss와 metrics에 집어 넣은 값으로 평가

x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)

print("loss :", loss)
print("acc :", acc)
print(np.around((y_pred)))
# print(y_pred)


