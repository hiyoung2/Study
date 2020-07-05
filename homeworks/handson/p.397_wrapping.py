# 10.3 신경망 하이퍼파라미터 튜닝하기
# 신경망의 유연성은 단점이기도
# 조정할 하이퍼파라미터가 많기 때문,,
# 아주 복잡한 네트워크 구조에서뿐만 아니라 간단한 다층 퍼셉트론에서도 층의 개수, 층마다 있는 뉴런의 개수,
# 각 층에서 사용할 활성화 함수, 가중치 초기화 전략 등 많은 것들을 바꿀 수 있다
# 어떻게 최적의 하이퍼파라미터 조합을 알 수 있을까

# 한 가지 방법은 많은 하이퍼파라미터 조합을 시도해보고 어떤 것이 검증 세트에서(또는 K-폴드 교차 검증으로) 가장 좋으 점수를 내는지 확인하는 것
# 예를 들어 GridSearchCV, RandomizedSearchCV를 사용해 하이퍼파라미터 공간을 탐색할 수 있다

# 이렇게 하려면! 케라스 모델을 사이킷런 추정기처럼 보이도록 바꿔야 한다!!!
# 먼저 일련의 하이퍼파라미터로 케라스 모델을 만들고 컴파일하는 함수를 만든다

import keras

def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 3e-3, input_shape = [8]) :
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden) :
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

# 이 함수는 주어진 입력 크기와 은닉층 개수, 뉴런 개수로(한 개의 출력 뉴런만 있는) 단변량 회귀를 위한 간단한 시퀀셜 모델을 만든다
# 그리고 지정된 학습률을 사용하는 SGD 옵티마이저로 모델을 컴파일
# 사이킷런과 마찬가지로 가능한 하이퍼파라미터에 적절한 기본값을 설정하는 것이 좋다

# build_model() 함수를 사용해 KerasRegrassor 클래스의 객체를 만든다

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# KerasRegressor 객체는 build_model() 함수로 만들어진 케라스 모델을 감싸는 간단한 래퍼 wraapper이다
# 이 객체를 만들 때 어떤 하이퍼파라미터도 지정하지 않았으므로 build_model()에 정의된 기본 하이퍼파라미터를 사용할 것
# 이제 일반적인 사이킷런 회귀 추정기처럼 이 객체를 사용할 수 있다
# 다음 코드와 같이 fit() 메서드로 모델을 훈련하고 score() 메서드로 평가하고 predict() 메서드로 예측을 만들 수 있다

# keras_reg.fit(X_train, y_train, epochs = 100, vlaidation_data = (X_valid, y_valid), callbacks= [keras.callbacks.EarlyStopping(patience = 10)])

# mse_test = keras_reg.score(X_test, y_test)

# y_pred = keras_reg.predict(X_new)

# fit() 메서드에 지정한 모든 매개변수는 케라스 모델로 전달된다
# 사이킷런은 손실이 아니라 점수를 계산하기 때문에(즉, 높을수록 좋다), 출력 점수는 음수의 MSE이다

# 모델 하나를 훈련하고 평가하는 것이 아니라 수백 개의 모델을 훈련하고 검증 세트에서 최상의 모델을 선택해야 한다
# 하이퍼파라미터가 많으므로 그리드탐색보다 랜덤탐색을 사용하는 것이 좋다
# 은닉층 개수, 뉴런 개수, 학습률을 사용해 하이퍼파라미터 탐색을 수행해보자

# import numpy as np

# from scipy.stats import reciprocal
# from sklearn.model_selection import RandomizedSearchCV

# param_distribs = {
#     "n_hidden" : [0, 1, 2, 3],
#     "n_neurons" : np.arange(1, 100),
#     "learning_rate" : reciprocal(3e-4, 3e-2)
# }

# rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter = 10, cv = 3)
# rnd_search_cv.fit(X_train, y_train, epochs = 100, validation_data = (X_valid, y_valid), callbacks = [keras.callbacks.EarlyStopping(pateince = 10)])

# RandomizedSearchCV는 k-겹 교차 검증을 사용하기 때문에 X_valid, y_valid를 사용하지 않는다
# 이 데이터는 조기 종료에만 사용된다

# 랜덤 탐색은 하드웨어와 데이터셋의 크기, 모델의 복잡도, n_iter, cv 매개변수에 따라 몇 시간이 걸릴 수 있다
# 실행이 끝나면 랜덤 탐색이 찾은 최상의 하이퍼파라미터와 훈련된 케라스 모델을 얻을 수 있다