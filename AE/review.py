# a 2345는 AUTO ENCODER가 아니다, 개념 설명을 위한 과정이었음

# 출처 : https://dataplay.tistory.com/34
# auto encoder 사용 이유
# 1. AutoEncoder 
# AutoEncoder를 사용하는 가장 주된 이유는 새로운 feature를 만들어 내는 것이다
# 인공신경망 때부터 유용하게 시용되다가, 딥러닝이 발전하면서 autoencoder도 같이 발전했다
# NLP의 기계 번역에서 주로 사용하는 seq2seq, 생성모델로 유명한 VAE, GAN 등을 알기 전에 autoencoder는 필수적으로 알아야하는 구조이다

# 2. AutoEncoder의 모델 구조
# AutoEncoder의 가장 큰 특징은 input과 output이 같다는 것
# input과 output이 같으면 도대체 모델이 무슨 용도가 있는가?
# autoencoder는 모델에 몇 가지 제약을 걸어줌으로써 모델이 의미가 있도록 만든다
# autoencoder의 모형을 보면,,,
# input과 output을 같도록 훈련시킨다
# input은 encoder를 통해서 z로 압축되고, 이 z를 다시 decoder를 통해서 x로 복원된다
# Input x와 Output x에 비해서 z의 크기는 작다
# Encoder는 점점 작아지고, Decoder는 점점 커진다
# 위와 같은 식으로 autoencoder는 input을 더 작은 feature로 압축을 했다가, 복원하는 과정으로 훈련을 한다
# 만약, 훈련이 너무 잘 되어서 input이 똑같이 복원되었다고 하자
# 그렇다면 압축된 feature인 z는 x보다 낮은 차원이지만, input x의 정보를 모두 가지고 있다고 볼 수 있다
# 이 z는 representation vector 또는 bottleneck 등의 이름으로 불린다
# 똑같은 정보를 가지고 있지만, 더 작은 차원이고, 다르게 표현된 feature라고 볼 수 있다
# 이런식으로 autoencoder는 압축 똔느 새로운 feature를 만드는 데 사용한다
# 이러한 특징으로 인해서 autoencoder를 과거부터 지금까지 유용하게 사용하고 있다

# 3. AutoEncoder의 발전
# AutoEncoder는 모델의 유용성 때문에 과거부터 이것저것 발전이 많았다
# Sparse Autoencoders라고 hidden layer의 일부 node만 사용하는 방법도 있고
# encoder와 decoder의 변수에 제약을 건다든지의 여러 가지 방법이 있다
# 눈 여겨 볼 만한 AE 몇 가지

# 1) Denoising AutoEncoder
# noise를 제거하기 위한 AE. 훈련을 진행할 때 의도적으로 input x에 noise를 add 하는 식으로 구성하는 게 많다
# 또는 input의 일부분을 의도적으로 제거하고 훈련을 한다
# 즉, input이 무언가 깔끔하지 않은 상태
# 이런 식으로 구성을 하면 이미지 또는 음성에 noise가 약간 끼어 있어도 깨끗한 이미지와 음성을 출력할 수 있다
# 논문에서만 사용하는 데이터가 아니라, 현실의 데이터를 사용하는 경우에는 데이터에 noise가 있는 경우가 상당히 많다
# 이런 경우에 denoising AE를 활용하려는 경우가 요즘에도 있다

# 2) Convolution & Deep AutoEncoder
# Autoencoder의 encoder와 decoder를 청므에는 Fc layer*를 이용해서 많이 구성했다면, 딥러닝이 발전하면서
# layer들을 CNN으로 구성하게 되고 layer의 수도 깊어지게 된다 -> 이러한 방식이 data의 feature를 뽑아내는 것에 도움을 주기 때문

# 3) Conditional AutoEncoder
# 딥러닝 모델을 학습할 때, 적절한 데이터(정보)의 추가는 항상 도움이 된다
# Mnist 데이터를 통해 원래의 이미지를 복원할 때, 기존에는 input x의 픽셀 값들만 주어졌다면,
# Conditional AE에서는 조건으로 input이 무슨 숫자인지, 정보를 준다
# x를 압축했다가 다시 복원하는 것에 있어서 숫자의 정보는 분명 도움이 될 것
# 이런 조건을 주는 경우에는 조건이 어떤 식으로 모델에 input이 되어야 하는가도 매우 중요한 문제이다
# 좋은 정보라고 할 지라도 모델을 학습하는 데에는 제대로 전달이 되지 않는 경우에는 성능의 향상이 일어나지 않는 법
# 가중치 학습을 할 때 정보를 주도록 잘 분배해야 한다
# DNN의 경우, 이미지를 flatten 한 다음에 숫자 정보를 concate 해준다든지
# CNN의 경우 이미지 채널을 늘린다든지, label을 어디에 첨가해주는 방식 등의 접근이 필요한다

# 3) VAE, 그리고 생성모델
# AE의 Decoder를 보면 input x를 압축해놓은 feature인 z를 다시 input x로 복원한다
# 그래서 decoder를 "생성 네트워크"라고도 부른다
# 잘 학습된 모델의 decoder만 분리해내서 새로운 z를 input으로 넣으면 새로운 output이 나올 것이니까!
# Mnist 데이터로 학습한 모델을 예시로 들면, 새롭게 그린 숫자가 나올 수도 있는 것

# z의 분포를 알 수만 있다면, 즉 mnist 데이터에서 숫자 0을 그릴 때 z가 어떻게 분포되어 있는지를 알고,
# 그 z 안에서 샘플링을 한다면 이제껏 없던 새로운 이미지가 등장한다
# 하지만, 기존의 AE에서는 이 z의 분포가 항상 달라진다
# 새로운 모델링을 할 때마다 달라지게 되고, 이런 경우에는 z를 사용하기가 쉽지 않다

# VAE는 이 점을 착안하여 z의 분포를 우리가 원하는 분포가 되도록 학습을 시킨다
# z의 분포가 항상 우리가 원하는, 알고 있는 분포로 매칭이 된다면 이 z를 이용하여 VAE라는 생성 모델을 만들어낼 수 있다
# 다만 이렇게 z를 우리가 원하는 분포로 바꾸는 과정에서 원래 훈련이 불가능한 상황이 일어난다
# 분포를 변환하는 과정이, 가중치 학습이 불가능하기 때문이다
# VAE 논문의 저자는 reparameterization trick 이라는 방식으로 문제를 해결함(추후 보충)

# 이외에도 VQ-VAE 등의 발전이 계속 있다
# 딥러닝을 포함한 머신러닝 기법은 대부분 예측을 하거나 분류를 하는 기능을 한다
# 그런데 AE는 '생성 모델'로서의 가능성이 존재한다
# 생성 모델이 제대로 작동한다면 지금보다 머신러닝이 하는 역할이 훨씬 커질 것이다
# 그래서 VAE, GAN 등의 생성 모델에 폭발적인 관심이 생겼고, 점점 발전하게 되었다


# FClayer : Fully Connected Layer
# 크게 보면 FClayer는 DNN에 포함되는 것
# 종종 DNN이라는 단어를 FC layer의 의미로 사용하는 경우가 있다




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

def order() :                         # (1) 항상 끝에는 colon :, 빠뜨리지 않도록 주의
    print("주문하실 음료를 알려주세요") # (2) 이 블록은
    drink = input()                   # 앞에서부터 네 칸씩
    print(drink,'주문하셨습니다.')     # 들여쓰기 한다

# order() # (3) 여기는 함수 실행 부분

# (1) 은 def문의 헤더 행, 함수 정의는 헤더 행에서부터 시작된다. 함수의 이름을 order라고 정의함
# (2) 는 함수의 본문 코드 블록. 이 코드는 함수가 호출 되었을 때 실행된다
#     몇 행이든 필요한 만큼 작성할 수 있다. 하지만 하나의 함수에서 너무 많은 일을 하지 않도록 주의1
#     함수의 내용이 너무 길면 내용을 파악하기 어려워진다

# (3) 은 함수를 호출하는 코드로, 함수 정의와는 별개다. def문으로 함수를 정의하는 것만으로는 함수의 내용이 실행되지 않는다.
#     함수를 실행시키려면 "호출"을 해야 한다!

# 연습문제
def print_absolute() :
    print("정수를 입력하세요")
    number = int(input())
    print("number의 절대값:", absolute(number))

print_absolute()

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