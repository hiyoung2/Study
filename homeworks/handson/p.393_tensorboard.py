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
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = 'relu', input_shape = X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss = "mean_squared_error", optimizer = keras.optimizers.SGD(lr = 0.05))

# 10.2.8 텐서보드를 사용해 시각화하기

# 텐서보드는 매우 좋은 interactive 시각화 도구이다
# 훈련하는 동안 학습 곡선을 그리거나 여러 실행 간의 학습 곡선을 비교하고 계산 그래프 시각화와 훈련 통계 분석을 수행할 수 있다
# 또한 모델이 생성한 이미지를 확인하거나 3D에 투영된 복잡한 다차원 데이터를 시각화하고 자동으로 클러스터링을 해주는 등 많은 기능을 제공한다
# 텐서보드는 텐서플로를 설치할 때 자동으로 설치되므로 이미 시스템에 준비되어 있다

# 텐서보드를 사용하려면 프로그램을 수정하여 이벤트 파일, event file 이라는 특별한 이진 로그 파일에 시각화하려는 데이터를 출력해야 한다
# 각각의 이진 데이터 레코드를 summary 라고 부른다
# 텐서보드 서버는 로그 디렉터리를 모니터링하고 자동으로 변경사항을 읽어 그래프를 업데이트 한다
# 훈련하는 중간에 학습 곡선 같이(약간의 지연은 있지만), 실시간 데이터를 시각화 할 수 있다
# 일반적으로 텐서보드 서버가 루트 root 로그 디렉터리를 가리키고 프로그램은 실행할 때마다 다른 서브디렉터리에 이벤트를 기록한다
# 이렇게 하면 복잡하지 않게 하나의 텐서보드 서버가 여러 번 실행한 프로그램의 결과를 시각화하고 비교할 수 있다

# 먼저 텐서보드 로그를 위해 사용할 루트 로그 디렉터리를 저으이
# 현재 날짜와 시간을 사용해 실행할 때마다 다른 서브디렉터리 경로를 생성하는 간단한 함수를 만듦
# 테스트는 하이퍼파라미터 값과 같은 추가적인 정볼르 로그 디렉터리 이름으로 사용할 수 있다
# 이렇게 하면 텐서보드에서 어떤 로그인지 구분하기 편리하다


import os 
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir() # 예를 들면, './my_logs/run_2019_06_07-15_15_22'


# [...] # 모델 구성과 컴파일
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid), callbacks = [tensorboard_cb])

