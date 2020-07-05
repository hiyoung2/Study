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