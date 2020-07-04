# input2, output2
# compile 단계에서 또 다른 parameter : loss_weights 추가
# loss_weights로 두 번째 모델인 분류 모델에 가중치 비율을 90% 주기

# 1. 데이터
import numpy as np

x1_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y1_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y2_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


# 2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input, concatenate


input1 = Input(shape = (1, ))

x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape = (1, ))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

x3 = Dense(100)(merge)
output1 = Dense(1)(x3)

x4 = Dense(70)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation = 'sigmoid')(x4)

model = Model(inputs = [input1, input2], outputs = [output1, output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], loss_weights = [0.1, 0.9], optimizer = 'adam', metrics = ['mse', 'acc'])

# loss_weights = [0.1, 0.9] : 첫 번째 모델에 가중치 비율 10%, 두 번째 모델에 가중치 비율 90%?

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs = 100, batch_size = 1)

# 4. 평가, 예측
loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train])
print("LOSS :", loss)


x1_pred = np.array([11, 12, 13, 14])
x2_pred = np.array([11, 12, 13, 14])


y_pred = model.predict([x1_pred, x2_pred])
print("y_pred :", y_pred)

# 결론
# 가중치 비율을 더 주어도 좋아지지 않음
# 각각 모델 별개로 돌리는 게 맞는 듯
# 분류모델 하나 분류모델지표 하나
# 회귀모델 하나 회귀모델지표 하나
# 각각 이렇게 돌리는 게 좋다

# 분류 회귀 동시에 돌리는 게 가능은 하다는 실습인 듯