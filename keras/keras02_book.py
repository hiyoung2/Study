# 일반적으로 node가 많고 layer가 깊을수록 더 잘 훈련을 함
# 하지만 어느 정도 훈련이 되면 중복 등의 문제로 인하여 훈련도가 오히려 떨어지는 경우도 발생
# 결국 인간은 딥러닝을 한다면 데이터를 준비할 때 x값과 y값을 준비하면 되고
# 모델을 생성할 때는 얼마나 많은 layer와 node를 준비할 것인지에 대해 설계해야 함

from keras.models import Sequential # 시퀀셜 모델(순차적 구성 모델) 만들기 위해 keras.models 에서 불러와야 함
from keras.layers import Dense      # 모델 구성하는데 Dense 레이어를 사용한다
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])  # 이 전체는 data set, 애초에 train data 와 tsest data가 따로 주어졌다(직접 split 할 필요가 없다)
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim =1, activation='relu'))   
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1, activation='relu'))

# model.summary()
"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 5)                 10           {1(input)+1(bias)} * 5(output) = 10 / (1*5)+(1*5) = (1+1)*5
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18           {5(input)+1(bias)} * 3(output) = 18 / (5*3)+(1*3) = (5+1)*3
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8            {3(input)+1(bias)} * 2(output) = 8 / (3*2)+(1*2) = (3+1)*2
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3            {2(input)+1(bias)} * 1(output) = 3 / (2*1)+(1*1) = (2+1)*1
=================================================================
Total params: 39
Trainable params: 39
Non-trainable params: 0

layer마다 weight 값이 계산됨 
input layer에서 다음 layer로 가면서 y=wx+b가 계산됨
첫 번째 input layer에서 다음 layer로 넘어가면서 b, 바로 bias값이 영향을 미침 (y = wx+b)
machine은 deeplearning을 할 때 bias도 1개의 node로 계산!!!!!!!!!

parameter = 연산의 갯수 (연산이 이루어지는 횟수?)
"""

# 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1) # validation_data = (x_train, y_train))

# 평가
loss, acc = model.evaluate(x_test, y_test, batch_size =1)
print("loss : ", loss)
print("acc : ", acc)

# 예측
output = model.predict(x_test)
print("결과물 : \n", output)





# accuracy 0 나올 수 있음
# Data가 너무 적거나 훈련량 적을 경우 충분히 나타날 수 있음

# y = wx+b
# h(x) = wx+b

# w : 최적의 가중치(weight)
# 가장 좋은 weight는 없음, 최적의 weight만 존재
# b는 bias
# h는 hypothesis

# x = 1, 2, 3 / y = 1, 2, 3 이라고 주어지는 data는 정제된 data
# if, x = 1, 2, 3, 4,    5
#     y = 1, 2, 3, 1000, 5
#                          이라는 data가 주어진다면?
# x = 4, y = 1000은 이상치(outlier)
# 이를 제거하는 과정이 data preprocessing(데이터 전처리)
# 데이터 전처리를 통해 우리는 정제된 data를 준비하면 된다.

# validation data?
# ML, DL에서 Data는 보통 train, validation, test 3가지로 나눔
# train data와 validation data는 training 과정에서 사용
# training 과정에서 model을 중간 중간 평가를 해야 함(이 model이 잘 학습 되는지 안 되는지)
# 즉, model의 성능을 중간에 평가
# why?
# 1. test accuracy를 가늠 해 볼 수 있음
#   machine learning의 목적은 unseen data 즉, test dat에 대해 좋은 성능을 내는 것
#   그러므로 model을 만든 후 이 model이 unseen data에 대해 얼마나 잘 동작할지에 대해 확인이 반드시 필요
#   하지만 training data를 사용해 성능을 평가하면 안 되기 때문에 따로 validation set을 만들어 정확도를 측정하는 것

# 2. model을 tuning하여 model의 성능을 높일 수 있음
#   예를 들어 overfitting(과적합) 등을 막을 수 있다
#   ex) training accuracy는 높은데 validation accuracy는 낮다면 data가 training set에서 overfitting이 일어났을 가능성 생각 해 볼 수 있음
#   그렇다면, overfitting을 막아서 training accuracy를 희생하더라도 validation accuracy와 training accuracy를 비슷하게 맞춰줄 필요가 있다
#   Deep Learning model을 구축한다면 regularization 과정 또는 epochs를 줄이는 등의 방식으로 overfitting을 막을 수 있음 

# 이 때 train data로 training 과정을 진행
# 그리고 training 과정에서 중간 중간 평가하는 것을 validation data가 해 준다
# train data와 validation data를 통해 훈련 된 이 model이 정답일까?
# NO
# 최종적으로 model이 '한 번도 보지 못한 data'에 대해서 평가를 해봐야 한다
# 그래야 이 model이 현실 세계, 우리의 서비스에서 실제로 잘 동작하는지를 판단 가능
# 이 때 사용하는 것이 test data(unseen data)
# 즉, test data는 model을 평가할 때 단 1번만 사용되며, 최종적으로 model의 평가 지표가 된다

# 보통, x_train, y_train : train data
# x_test, y_test : validation data
