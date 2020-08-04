from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import random


# autoencoder 함수 정의
# 함수 정의 : def, return 

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784, ), activation  = 'relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

x_train = x_train / 255
x_test = x_test / 255

print("x_train.reshape :", x_train.shape)
print("x_test.reshape :", x_test.shape)


# 2. 모델 구성 (함수로 만들어 놓은 모델 사용)
# model = autoencoder(hidden_layer_size=154)
model_01 = autoencoder(1)
model_02 = autoencoder(2)
model_04 = autoencoder(4)
model_08 = autoencoder(8)
model_16 = autoencoder(16)
model_32 = autoencoder(32)

# model.summary()


# 3. 컴파일, 훈련
model_01.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_01.fit(x_train, x_train, epochs=10) 

model_02.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_02.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_02.fit(x_train, x_train, epochs=10) 

model_04.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_04.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_04.fit(x_train, x_train, epochs=10) 

model_08.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_08.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_08.fit(x_train, x_train, epochs=10) 

model_16.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_16.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_16.fit(x_train, x_train, epochs=10) 

model_32.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# model_32.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
model_32.fit(x_train, x_train, epochs=10) 


# 4. 예측
output_01 = model_01.predict(x_test)

output_02 = model_02.predict(x_test)

output_04 = model_04.predict(x_test)

output_08 = model_08.predict(x_test)

output_16 = model_16.predict(x_test)

output_32 = model_32.predict(x_test)

# print(output_01.shape[0]) # 1000.


# 그림을 그리자!!!
fig, axes = plt.subplots(7, 5, figsize = (15, 15)) # rows = 7, cols = 5
random_imgs = random.sample(range(output_01.shape[0]), 5)

# 출처 : https://jinseok12.tistory.com/19
# random.sample(range(시작, 종료), 리턴값 개수) : 시작값 이상 종료값 미만의 값을 리스트 형식으로 반환(중복은 없다)
# x_text의 예측값 10000개 중에서 5개가 무작위로 random_imgs라는 변수에 리스트 형태로 반환된다?

# Python 9. range 함수, Random 모듈
# Range 함수
# for문은 숫자 리스트를 자동으로 만들어주는 range 함수와 함께 사용되는 경우가 많다
# range 함수는 기본적으로
# range(종료) : 0부터 종료숫자 -1까지 범위
# range(시작, 종료) : 시작 숫자부터 종료 숫자 -1 까지 숫자 범위
# range(시작, 종료, 증가값) : 시작 숫자부터 종료 숫자 -1까지 증가값만큼 증가된 숫자범위
# ex. list(range(1, 10, 2)) : 1 이상 9 이하의 범위에서 1부터 2씩 더해진 값

# Random 모듈
# 파이썬에서 random 관련 함수들을 사용할 수 있도록 random 모듈을 내장하고 있다
# random 함수를 통하여 난수를 생성할 수 있다
# random 함수들을 사용하기 위해서는 먼저 import random이 필요하다

# random.randint(시작, 종료): 시작값 이상 종료값 이하의 정수를 난수로 생성

# random.randrange(시작, 종료, 증가값) : 시작값 이상 종료값 미만의 난수를 반환

# random.sample(range(시작, 종료), 리턴값 개수) : 시작값 이상 종료값 미만의 값을 리스트 형식으로 반환(중복은 없다)

# random.choice(변수명) : 리스트, 튜플에서 랜덤하게 항목을 뽑아낼 때 사용한다

# random.shuffle(변수명) : 리스트에 있는 항목들의 순서를 랜덤하게 재배치한다
# 리턴값이 없고, 전달하는 변수 자체를 바꿔버린다(리스트가 바뀌기 때문에, 튜플은 안 된다)

outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes) :
    for col_num, ax in enumerate(row) :
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
