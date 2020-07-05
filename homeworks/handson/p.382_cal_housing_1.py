# 10.2.3 시퀀셜 API를 사용하여 회귀용 다층 퍼셉트론 만들기
# 캘리포니아 주택 가격 데이터셋으로 바꿔 회귀 신경망으로 이를 해결
# 간편하게 사이킷런의 fetch_california_housing()함수를 이용, 데이터를 적재

# 이 데이터셋은 수치 특성만 있음
# 누락된 데이터 x
# 데이터를 불러온 후 훈련 세트, 검증 세트, 테스트 세트로 나누고 모든 특성의 스케일을 조정

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

# 시퀀셜 API를 사용해 회귀용 MLP를 구축, 훈련, 평가, 예측하는 방법은 분류에서 했던 것과 매우 비슷
# 주된 차이점은 출력층이 활성화함수가 없는 하나의 뉴런(하나의 값을 예측하기 때문)을 가진다는 것과
# 손실 함수로 평균 제곱 오차를 사용한다는 것
# 현재 이 데이터셋에는 잡음이 많기 때문에 과대적합을 막는 용도로 뉴런 수가 적은 은닉층 하나만 사용한다

import keras
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = 'relu', input_shape = X_train.shape[1:]),
    keras.layers.Dense(1)
])

# print("X_train.shape[1:]", X_train.shape[1:]) # (8,)

model.compile(loss = "mean_squared_error", optimizer = "sgd")

history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid))

mse_test = model.evaluate(X_test, y_test)
print("MSE_TEST :", mse_test)
# MSE_TEST : 0.3867035560367643

X_new = X_test[:3]
y_pred = model.predict(X_new)
print("y_pred :", y_pred)
'''
y_pred : [[1.5293437]
 [2.7570617]
 [0.8958824]]
'''

# 시퀀셜 API는 사용하기 아주 쉽다
# Sequential 모델이 매우 널리 사용되지만 입력과 출력이 여러 개거나 더 복잡한 네트워크 토폴로지를 갖는 신경망을 만들어야 할 때에는,,
# 케라스에서 제공하는 함수형 API를 사용해야 한다

