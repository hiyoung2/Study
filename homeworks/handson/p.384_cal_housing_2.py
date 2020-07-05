# 10.2.4 함수형 API를 사용해 복잡한 모델 만들기
# 순차적이지 않은 신경망의 한 예는 wide & deep 신경망이다

# 이 신경망 구조는 2016년 헝쯔 청의 논문에서 소개됨
# 입력의 일부 또는 전체가 출력층에 바로 연결된다
# 이 구조를 사용하면 신경망이 (깊게 쌓은 층을 사용한)복잡한 패턴과 (짧은 경로를 사용한) 간단한 규칙을 모두 학습할 수 있다

# 이와는 대조적으로 일반적인 MLP는 네트워크에 있는 층 전체에 모든 데이터를 통과시킨다
# 데이터에 있는 간단한 패턴이 연속된 변환으로 인해 왜곡될 수 있다

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full
)

print("X_train.shape :", X_train.shape) # (11610, 8)
print("X_train.shape[1:]", X_train.shape[1:]) # (8,)
print("y_train.shape :", y_train.shape) # (11610,)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

import keras

# input_ = keras.layers.Input(shape = X_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation = 'relu')(input_)
# hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
# concat = keras.layers.Concatenate()([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.Model(inputs = [input_], outputs = [output])

# 먼저 Input 객체를 만들어야 한다, 이 객체는 shape와 dtype을 포함하여 모델의 입력을 정의한다
# 한 모델은 여러 개의 입력을 가질 수 있다

# 30개의 뉴런과 ReLU 활성화 함수를 가진 Dense 층을 만든다, 이 층은 만들어지자마자 입력과 함께 함수처럼 호출된다
# 이를 함수형 API라고 부르는 이유이다
# 케라스에 층이 연결될 방법을 알려주었을 뿐, 아직 어떠한 데이터도 처리하지 않은 상태

# 두 번째 은닉층을 만들고 함수처럼 호출한다, 첫 번째 층의 출력을 전달한 점에 주의!
# Concatenate 층을 만들고 또 다시 함수처럼 호출하여 두 번째 은닉층의 출력과 입력을 연결한다
# keras.layers.concatenate() 함수를 사용할 수도 있다
# 이 함수는 Concatenate 층을 만들고 주어진 입력으로 바로 호출한다
# 하나의 뉴런과 활성화 함수가 없는 출력층을 만들고 Concatenate 층이 만든 결과를 사용해 호출한다
# 마지막으로 사용할 입력과 출력을 지정하여 케라스 model을 만든다

# 모델을 컴파일한 다음, 훈련, 평가, 예측을 수행하면 된다

# 만약 일부 특성은 짧은 경로로 전달, 다른 특성들은 깊은 경로로 전달하고 싶다면?
# 이 경우 한 가지 방법은 여러 입력을 사용하는 것
# 예를 들어 5개 특성(특성 인덱스 0~4)을 짧은 경로로 보내고, 6개 특성(2~7)은 깊은 경로로 보낸다고 가정

input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = 'relu')(input_B)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name = 'outpupt')(concat)

model = keras.Model(inputs = [input_A, input_B], outputs = [output])

# 이렇게 모델이 복잡해지면 적어도 가장 중요한 층에는 이름을 붙이는 것이 좋다
# 모델을 만들 때 inputs = [inoput_A, input_B]와 같이 지정했다
# 모델 컴파일은 이전과 동일하지만, fit() 메서드를 호출할 때 하나의 입력 행렬 X_train을 전달하는 것이 아니라
# 입력마다 하나씩 행렬의 튜플(X_train_A, X_train_B)을 전달해야 한다
# X_valid에도 동일 적용
# evaluate()나 predict()를 호출할 대 X_test, X_new에도 동일

# 컴파일, 훈련
model.compile(loss = 'mse', optimizer = keras.optimizers.SGD(lr = 1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit([X_train_A, X_train_B], y_train, epochs = 20, validation_data = ([X_valid_A, X_valid_B], y_valid))


# 평가, 예측
mse_test = model.evaluate([X_test_A, X_test_B], y_test)
print("MSE_TEST :", mse_test)
# MSE_TEST : 0.4390312316805817

y_pred = model.predict([X_new_A, X_new_B])
print("Y_PRED :", y_pred)

'''
Y_PRED : [[0.9053252]
 [1.7516527]
 [1.6748521]]
'''