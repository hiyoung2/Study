# 10.2.2 시퀀셜 API를 사용하여 이미지 분류기 만들기

# 케라스를 사용하여 데이터셋 적재하기

import tensorflow as tf
from tensorflow import keras

# print("tensorflow version :", tf.__version__)
# print("keras version :", keras.__version__)

# tensorflow version : 2.0.0
# keras version : 2.2.4-tf

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# 사이킷런 대신 케라스를 사용하여 MNIST나 fashionMNIST 데이터를 적재할 때 중요한 차이점은
# 각 이미지가 784 크기의 1D 배열이 아니라 28 x 28 크기의 배열이라는 것, 그리고
# 픽셀 강도가 실수(0.0 ~ 255.0)가 아니라 정수 (0 ~ 255)로 표현되어 있다

print("X_train_full.shape :", X_train_full.shape) # (60000, 28, 28)
print("X_train_full.dtype :", X_train_full.dtype) # uint8

# 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나눠져 있음
# 하지만 검증 세트가 없으므로 여기에서 만듦
# 또한 경사하강법으로 신경망을 훈련하기 때문에 입력 특성의 스케일을 조정해야 한다
# 간편하게 픽셀 강도를 255.0으로 나누어 0 ~ 1 사이 범위로 조정!(자동으로 실수로 변환된다)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# MNIST의 레이블이 5이면 이 이미지가 손글씨 숫자 5를 나타낸다
# 그러나 패션 MNIST는 레이블에 해당하는 아이템을 나타내기 위해 클래스 이름의 리스트를 만들어야 한다

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]]) # Coat


# 시퀀셜 API를 사용하여 모델 만들기
# 신경망을 만들보자
# 다음은 두 개의 은닉층으로 이루어진 분류용 다층 퍼셉트론이다

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

# 위와 같이 층을 하나씩 추가하지 않고 Sequential 모델을 만들 때 층의 리스트를 전달할 수 있다
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape = [28, 28]),
#     keras.layers.Dense(300, activation = 'relu'),
#     keras.layers.Dense(100, activation = 'relu'),
#     keras.layers.Dense(10, activation = 'softmax')
# ])

# activation = 'relu' 로 지정하는 것과
# activation = keras.activations.relu 로 지정하는 것은 동일!


################## keras.io 의 예제 코드 사용하기 ################## 
# keras.io 문서에 있는 예제 코드는 tf.keras에서 잘 작동한다
# 하지만 import 명령을 수정해야 한다
# 예를 들어 keras.io 에 다음과 같은 예제가 있다
# from keras.layers import Dense
# output_layer = Dense(10)

# 이 import 명령을 다음과 같이 바꿔야 한다
# from tensorflow.keras.layers import Dense
# output_layer = Dense(10)

# 원한다면 전체 경로를 사용할 수 있다
# from tensorflow import keras
# output_layer = keras.layers.Dense(10)

# 이 방식이 조금 장황하지만 어떤 패키지를 사용하는지 쉽게 알 수 있고 표준 클래스와 사용자 정의 클래스 사이에서 혼란을 피할 수 있다
############################################################################################################################## 



# 모델의 summary() 메서드는 모델에 있는 모든 층을 출력한다
# 각 층의 이름(층을 만들 때 지정하지 않으면 자동으로 생성), 출력 크기(None은 배치 크기에 어떤 값도 가능하다는 뜻), 파라미터 개수가 함께 출력된다
# 마지막에는 훈련되는 파라미터와 훈련되지 않은 파라미터를 포함하여 전체 파라미터 개수를 출력한다
# 지금 예제에서는 훈련되는 파라미터만 있다

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 300)               235500
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
'''

# Dense 층은 보통 많은 파라미터를 가진다
# 첫 번째 은닉층은 784 * 300 개의 연결 가중치와 300개의 편향을 가진다
# 이를 더하면 파라미터가 235,500개나 된다
# 이런 모델은 훈련 데이터를 학습하기 충분한 유연성을 가진다
# 또한 과대적합의 위험을 갖는다는 의미이기도 하다
# 특히 훈련 데이터가 많지 않을 경우에 그렇다
# 
# 모델에 있는 층의 리스트를 출력하거나 인덱스로 층을 쉽게 선택할 수 있다
# 또는 이름으로 층을 선택할 수도 있다

print(model.layers)
# [<tensorflow.python.keras.layers.core.Flatten object at 0x00000241161FB548>, <tensorflow.python.keras.layers.core.Dense object at 0x000002411BEA5D88>, 
# <tensorflow.python.keras.layers.core.Dense object at 0x000002411BEA5B88>, <tensorflow.python.keras.layers.core.Dense object at 0x00000241140763C8>]

hidden1 = model.layers[1]
print(hidden1.name) # dense

print(model.get_layer('dense') is hidden1) # True

# 층의 모든 파라미터는 get_weights() 메서드와 set_weights() 메서드를 사용해 접근할 수 있다
# Dense 층의 경우에는 연결 가중치와 편향이 모두 포함되어 있다

weights, biases = hidden1.get_weights()
print(weights)
'''
[[-0.04804367 -0.05910118  0.03597022 ... -0.06098216 -0.05717362
  -0.02862689]
 [-0.06574248  0.00326937  0.02417129 ... -0.04956041 -0.05397364
   0.01969548]
 [ 0.00695838  0.02074526 -0.06670496 ...  0.0601702  -0.04438786
   0.0524326 ]
 ...
 [-0.02453262  0.01663984  0.02661978 ...  0.04317413  0.02868603
   0.00998875]
 [ 0.07364933 -0.06110699  0.0060849  ...  0.07199687  0.01150034
   0.00820074]
 [-0.01424525 -0.0551304   0.01921874 ...  0.00794289 -0.0403213
   0.04778106]]
'''
print("weights.shape :", weights.shape) # (784, 300)


print(biases)
'''
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
'''
print("biases.shape :", biases.shape)  # (300,)  

# Dense 층은 연결 가중치를 무작위로 초기화한다
# 편향은 0으로 초기화한다
# 다른 초기화 방법을 사용하고 싶다면 층을 만들 때 kernel_initializere(kernel은 연결 가중치 행렬의 또 다른 이름)와
# bias_initializer 매개 변수를 설정할 수 있다

# 모델 컴파일
# 모델을 만들고 나서 compile() 메서드를 호출하여 사용할 손실 함수와 옵티마이저optimizer를 지정해야 한다
# 부가적으로 훈련과 평가 시에 계산할 지표를 추가로 지정할 수 있다

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# loss = 'sparse_categorical_crossentropy' == loss = keras.losses.spares_categorical_crossentropy
# optimizer = 'sgd' == optimizer = keras.optimizers.SGD()
# metrics = ['accuracy] == metrics = [keras.metrics.spares_categorical_accuracy]

# SGD 옵티마이저를 사용할 때에는 학습률을 튜닝하는 것이 중요하다
# 따라서 보통 optimizer = keras.optimizers.SGD(lr = ?)와 같이 사용하여 학습률을 지정한다
# optimizer = 'sgd' 는 기본값 lr = 0.01(default) 을 사용
# lr 매개변수는 하위 호환성(텐서플로 1.13버전 이하)를 위해 쓰는 것, lr 대신 learning_rate를 사용하라


# 모델 훈련과 평가
# 모델을 훈련하려면 간단하게 fit() 메서드를 호출한다


history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

# 입력 특성(X_train)과 타깃 클래스(y_train), 훈련할 에포크 횟수(지정하지 않으면 기본값이 1이라서 좋은 솔루션으로 수렴하기 충분하지 않을 것)를 전달
# 검증 세트도 전달(이는 선택 사항)
# 케라스는 에포크가 끝날 때마다 검증 세트를 사용해 손실과 추가적인 측정 지표를 계산한다
# 이 지표는 모델이 얼마나 잘 수행되는지 확인하는 데 유용하다
# 훈련 세트 성능이 검증 세트보다 월등히 높다면 아마도 모델이 훈련 세트에 과대적합되었을 것
# (또는 버그가 있을 수도 있다, 가령 훈련 세트와 검증 세트 간의 데이터가 올바르지 않을 경우)

# 끝!, 신경망을 훈련했다
# 훈련 에포크마다 케라스는 (진행 표시줄, progress bar 과 함께) 처리한 샘플 개수와 샘플마다 걸린 평균 훈련 시간,
# 훈련 세트와 검증 세트에 대한 손실과 정확도(또는 추가로 요청한 다른 지표)를 출력한다
# 훈련 손실이 감소한다? -> 좋은 신호

'''
Epoch 28/30
55000/55000 [==============================] - 3s 61us/sample - loss: 0.2350 - accuracy: 0.9154 - val_loss: 0.2981 - val_accuracy: 0.8906
Epoch 29/30
55000/55000 [==============================] - 3s 61us/sample - loss: 0.2306 - accuracy: 0.9165 - val_loss: 0.2994 - val_accuracy: 0.8928
Epoch 30/30
55000/55000 [==============================] - 3s 60us/sample - loss: 0.2268 - accuracy: 0.9192 - val_loss: 0.3008 - val_accuracy: 0.8952
'''
# 30번 에포크 이후 검증 정확도가 89%
# 훈련 정확도와 차이가 크지 않기 때문에 과대적합이 많이 일어나지는 않은 듯 보임

# TIP
# validation_data 매개변수에 검증 세트를 전달하는 대신 케라스가 검증에 사용할 훈련 세트의 비율을 지정할 수 있다
# validation_split = 0.1로 쓰면 케라스는 검증에 (섞기 전의) 데이터의 마지막 10%를 사용한다

# 어떤 클래스는 많이 등장하고 다른 클래스는 조금 등장하여 훈련 세트가 편중되어 있다면
# fit() 메서드를 호출할 때 class_weight 매개변수를 지정하는 것이 좋다
# 적게 등장하는 클래스는 높은 가중치를 부여하고 많이 등장하는 클래스는 낮은 가중치를 부여한다
# 케라스가 손실을 계산할 때 이 가중치를 사용한다
# 샘플별로 가중치를 부여하고 싶다면 sample_weight 매개 변수를 지정한다
# (class_weight 와 sample_weight 가 모두 지정되면 케라스는 두 값을 곱하여 사용한다)
# 어떤 샘플은 전문가에 의해 레이블이 할당되고 다른 샘플은 크라우드소싱(crowdsourcing) 플랫폼을 사용해 레이블이 할당되었다면
# 샘플별 가중치가 도움될 수 있다, 아마 전자에 더 높은 가중치를 부여할 것
# validation_data 튜플의 세 번째 원소로 검증 세트에 대한 샘플별 가중치를 지정할 수도 있다
# (클래스 가중치는 지정하지 못한다)

# fit() 메서드가 반환하는 history 객체에는 훈련 파라미터(history.params), 수행된 에포크 리스트(history.epoch)가 포함된다
# 이 객체의 가장 중요한 속성은 에포크가 끝날 때마다 훈련 세트와 (있다면) 검증 세트에 대한 손실과 측정한 지표를 담은
# 딕셔너리(history.history)이다!
# 이 딕셔너리를 사용해 판다스 데이터 프레임을 만들고 plot() 메서드를 호출하면 학습 곡선(learning curve)를 볼 수 있다

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # 수직축의 범위를 [0-1] 사이로 설정
# plt.show()


# 현재 이 모델의 학습 곡선을 보면,,,
# 훈련하는 동안 훈련 정확도와 검증 정확도가 꾸준히 상승
# 반면 훈련 손실과 검증 손실은 감소 -> 좋은 모델
# 또한 검증 곡선과 훈련 곡선이 가깝다 -> 크게 과대적합되지 않았다는 증거
# 검증 손실은 에포크가 끝난 후에 계산되고, 훈련 손실은 에포크가 진행되는 동안에 계산된다
# 따라서 훈련 곡선은 에포크의 절반만큼 왼쪽으로 이동해야 한다
# 그렇게 보면 훈련 초기에 훈련 곡선과 검증 곡선이 거의 완벽하게 일치한다

# TIP : 훈련 곡선을 볼 때 왼쪽으로 에포크의 절반만큼 이동해서 생각!

# 일반적으로 충분히 오래 훈련하면 훈련 세트의 성능이 검증 세트의 성능을 앞지른다
# 검증 손실이 여전히 감소한다면 모델이 아직 완전히 수렴되지 않았다고 볼 수 있다
# 훈련을 계속해야 할 것

# 케라스에서는 fit() 메서드를 다시 호출하면 중지되었던 곳에서부터 훈련을 이어갈 수 있다

# 모델 성능에 만족스럽지 않다면 처음으로 되돌아가서 하이퍼파라미터 튜닝을 해야 한다
# 가장 청므에 확인할 것은 학습률
# 학습률이 도움이 되지 않으면 다른 optimizer를 테스트해봐라
# 항상 다른 하이퍼파라미터를 바꾼 후에는 학습률을 다시 튜닝해야 한다

# 그래도 여전히 성능이 높지 않으면 층 개수, 층에 있는 뉴런 개수, 은닉층이 사용하는 활성화 함수와 같은 모델의 하이퍼파라미터 튜닝을 해 봐라
# 배치 크기와 같은 다른 하이퍼파라미터를 튜닝해 볼 수도 있다
# fit() 메서드를 호출할 때 batch_siae 매개변수로 지정함, default = 32


# 모델의 검증 정확도가 만족스럽다면 모델을 상용 환경으로 배포하기 전에 테스트 세트로 모델을 평가하여 일반화 오차를 추정해야 한다
# 이 때 evaluate() 메서드를 사용한다
# (이 메서드는 batch_size와 saple_weight 같은 다른 매개변수도 지원한다)

model.evaluate(X_test, y_test)
# loss: 0.2261 - accuracy: 0.8769

# 검증 세트보다 테스트 세트에서 성능이 조금 낮은 것이 일반적
# 하이퍼파라미터를 튜이한 곳이 테스트 세트가 아니라 검증 세트이기 때문이다
# 테스트 세트에서 하이퍼파라미터를 튜닝하려는 유혹을 참아야 한다
# 그렇지 않으면 일반화 오차를 매우 낙관적으로 추정하게 된다


# 모델을 사용해 예측을 만들기
# 모델의 prerdict() 메서드를 사용해 새로운 샘플에 대해 예측을 만들 수 있다
# 현재 예에서는 실제로 새로운 샘플이 없기 때문에 테스트 세트의 처음 3개 샘플을 사용

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
'''
[[0.   0.   0.   0.   0.   0.   0.   0.04 0.   0.95]
 [0.   0.   0.99 0.   0.01 0.   0.   0.   0.   0.  ]
 [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
'''
# 각 샘플에 대해 0에서 9까지의 클래스마다 각각의 확률을 모델이 추정함
# 예를 들어, 첫 번째 이미지에 대해서는 클래스 클래스 9(앵클 부츠)의 확률을 95%, 클래스 7일 확률 4%로 추정

# 가장 높은 확률을 가진 클래스에만 관심이 있다면 predict_classes() 메서드를 사용할 수 있다
# argmax 쓸 필요 없네

import numpy as np

y_pred = model.predict_classes(X_new)
print(y_pred)
# [9 2 1]

print(np.array(class_names)[y_pred])
# ['Ankle boot' 'Pullover' 'Trouser']

y_new = y_test[:3]
print(y_new)
# [9 2 1]