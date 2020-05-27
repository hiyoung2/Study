import numpy as np  # data를 넣기 위한 배열로 numpy 사용
#
#  데이터 생성
x = np.array([1,2,3,4,5,6,7,8,9,10])    
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 컴퓨터에 1 입력 1 출력, 2 입력 2 출력, ... 10 입력 10 출력되는 구조 
# 머신러닝에서는 보통 '트레이닝(training)' 시킨다고 함

# keras를 사용할 수 있는 환경 구축
from keras.models import Sequential
from keras.layers import Dense

# 딥러닝 모델의 완성
model = Sequential()                                    # 딥러닝 모델을 순차적으로 구성하겠다는 뜻
model.add(Dense(1, input_dim=1, activation='relu'))     # 순차적 구성 모델에 Dense 레이어(layer)를 추가하겠다는 의미
                                                        # dim = dimension (차원), dim=1 : 1차원
                                                        # if, 삼성전자 주가 예측시에는 주가 data, 환율 등 여러가지 input data가 존재
                                                        # 5가지 종류의 data가 있다면 이 때는 input_dim=5 가 됨


# loss : 손실 함수는 어떤 것을 사용할 것인가? mean_squared_error 평균제곱법을 사용하겠다
# optimizer : 최적화 함수는? adam 옵티마이저를 사용하겠다
# metrcis : 어떤 방식? accuracy(정확도)로 판정하겠다
# 위와 같은 옵션으로 compile한다

# 컴파일
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])

# keras의 모델 실행은 'fit'
# epochs : 훈련 횟수
# 훈련 횟수가 많아질수록 아마 좋은 결과가 나올 것이다. 하지만 너무 정말 심하게 지나치다면??

# batch_size : 몇 개씩 끊어서 작업 할 것인가
# 여기에서는 10개의 data를 1개씩 잘라서 작업하게 되므로, 1씩임
# batch_size를 크게 잡을 경우 속도는 빨라지지만 정확도가 떨어짐
#       반대로 작게 잡을 경우 속도는 떨어지지만 정확도는 올라감

# 물론, 너무 많은 데이터에 너무 작은 batch_size를 줄 경우 오히려 정확도가 떨어질 수 있음
# data가 5만 개인데 batch_size = 1, 하나하나씩 모두 끊어서 작업한다면?
# batch_size : 일괄작업 크기 / batch(일괄)
# batch_size 1이고 epochs 1이면 data별로 10번 훈련, 전체 data set은 1번 훈련하는 것
# batch_size 5이고 epochs 1이면 각 데이터는 5개씩 묶어져 2그룹으로 2번 훈련, 전체 data set은 1번 훈련
# ex) x {1,2,3,4,5} {6,7,8,9,10}
#     y {1,2,3,4,5} {6,7,8,9,10}

"""
loss :  1.839634137468238e-07
acc :  1.0
"""

# overfiting(과적합)의 영향 때문 (추후 설명)

# 훈련
model.fit(x, y, epochs= 100, batch_size=1)  # fit : 모델 실행, 모델 학습, 훈련 시킴

# 평가
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)



# 위와 같은 구조를 수학식으로 풀면 y = ax + b, 1차 함수형식임
# y = wx + b
# h(x) = wx + b
# 딥러닝에서는 a를 w, weight(가중치)로 바꿈 / 최적의 가중치!!!
# b는 'bias', h는 'hypothesis'
# 딥러닝은 예측(predict)를 하기 위한 것이기 때문에 hypothesis(가설)의 h를 사용한다고 이해하면 됨
# 결국 딥러닝은 우리가 빅데이터 등으로 준비한 x값(입력값)과 y값(결괏값)을 가지고 
# 컴퓨터에 훈련(train)을 시켜서 w값(weight, 가중치)과 b값(bias, 절편)을 구하는 행위의 반복
# 이 때 컴퓨터는 한 가지 값을 더 제공하게 되는데, 이것이 Cost(비용)임
# 비용인 Cost값은 낮을수록 좋음
# 이후 정확한 값이 예측되었는지 확인하기 위해 'accuracy'와 'predict'를 사용
# 딥러닝은 결국 1차 함수라는 사실을 명심


