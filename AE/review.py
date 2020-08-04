# a 2345는 AUTO ENCODER가 아니다, 개념 설명을 위한 과정이었음

# keras 2.4.0 version이 최근 release 되었다
# 더이상 multi backend를 지원하지 않는다
# 2.4.0 version은 기존 구현 코드를 모두 삭제하고 대신 tensorflow.keras로 redirection 한다

# 일단 나는 현재 2.3.1 version 사용중

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# autoencoder 함수 정의
# 함수 정의 : def, return 


# 출처 : https://python.bakyeono.net/chapter-3-3.html 

# 파이썬에서 함수를 정의할 때는 def 문을 사용한다
# def : 정의하다, define 
# def문의 양식
# def 함수이름() : (1)
#     본문        (2)
# (1) : def 예약어로 시작하는 첫 행에는 함수의 이름을 쓴다. 함수의 이름은 의미를 알 수 있게 짓는다. 함수 이름 뒤에는 괄호를 붙인다.
# (2) : 함수의 본문은 함수를 호출했을 때 실행할 파이썬 코드이다. 원하는 만큼 여러 행의 코드를 작성할 수 있으며
#       각 행의 앞에 띄어쓰기를 네 번씩 해야 한다.(Tab) == 들여쓰기
#       들여쓰기는 코드의 블록(구역)을 형성한다. 나란히 들여쓰기 된 코드 블록은 하나의 def문 안에 포함된 코드임을 나타낸다
#       들여쓰기가 끝나면, 그 함수의 정의가 끝난다.

# 실습
# def문으로 함수 정의하기
# 사용자로부터 이름을 입력받고 인사를 출력하는 함수를 정의해보자

def order() :                         # 항상 끝에는 colon :, 빠뜨리지 않도록 주의
    print("주문하실 음료를 알려주세요") # 이 블록은
    drink = input()                   # 앞에서부터 네 칸씩
    print(drink,'주문하셨습니다.')     # 들여쓰기 한다

order() # 여기는 함수 실행 부분



'''
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
model = autoencoder(hidden_layer_size=154)
# model = autoencoder (32)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # loss = mse 
# loss: 0.0102 - acc: 0.0110(32)
# loss: 0.0014 - acc: 0.0163 (154)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # loss = binary_crossentropy 
# loss: 0.0938 - acc: 0.8141 (32)
# loss: 0.0657 - acc: 0.8155 (154)


# mse, binary_crossentropy 둘 다 적용 시켜 봐야 한다

model.fit(x_train, x_train, epochs=10) 

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
'''