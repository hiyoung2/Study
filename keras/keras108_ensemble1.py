# 분류와 회귀를 동시에?
# ex) x : 우리 반 학생들의 키
#     y1 : 키, y2 : 성별

# 결론은? 별로

# 1. 데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y1_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y2_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape= (1, ))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)
output1 = Dense(1)(x2)
# outpu1 layer = dense_5

x3 = Dense(70)(x1)
x3 = Dense(70)(x3)
output2 = Dense(1, activation = 'sigmoid')(x3)
# output2 layer = dense_8

model = Model(inputs = input1, outputs = [output1, output2])
model.summary()


# 3. 컴파일, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer = 'adam', metrics = ['mse', 'acc'])
model.fit(x_train, [y1_train, y2_train], epochs = 100, batch_size = 1)

# 4. 평가, 에측
loss = model.evaluate(x_train, [y1_train ,y2_train])
print("LOSS :", loss)

# loss: 0.6920 - dense_5_loss: 6.5269e-04 - dense_8_loss: 0.6914 - dense_5_mse: 6.5269e-04 - dense_5_acc: 1.0000 - dense_8_mse: 0.2492 - dense_8_acc: 0.6000
# loss = dense_5_loss + dens_8_loss

x1_pred = np.array([11, 12, 13, 14])
y_pred = model.predict(x1_pred)
print("y_pred :", y_pred)