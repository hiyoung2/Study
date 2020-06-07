# keras 모델로 바꿔라
# input, output layer만 둬야 한다, hidden은 nope

# from sklearn.svm import LinearSVC, SVC
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0] # xor

# 딥러닝에서는 기존의 데이터로는 문제가 생긴다
# 딥러닝 연산 : 가중치의 곱셈, w와 x가 곱해짐, bias는 더해지고
# 케라스에서는 행렬 연산, 행렬 곱셈을 하기 위해 넘파이를 쓴다
# 딥러닝은 각 레이어마다 가중치 연산을 하므로 넘파이로 바꿔줘야 한다

# 리스트에서는 덧셈이 되면 [0, 0, 1, 0]이 됨
# 리스트 형태는 그냥 append 될 뿐, 연산 자체가 이뤄지지 않는다
# 머신러닝에서는 가중치 연산이 이뤄지지 않는다 따라서 그냥 리스트도 가능
# 문자 형태들도 그냥 수치로 바꾼다 -> "label encoder" 를 통해서
# "label encoder"에 dataset을 넣으면 사람 : 1, 고양이 : 2 이런 방식으로 알아서 숫자로 바꿔줌


import numpy as np
x = np.array(x_data)
y = np.array(y_data)

print("x : ", x)
print("y : ", y)

print("x.shape : ", x.shape) # (4, 2)
print("y.shape : ", y.shape) # (4,) / 스칼라4, 벡터1, output dimension = 1

# 2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1) 

# lin = LinearSVC()
# sv = SVC()
# kn = KNeighborClassifier
# 이런 식으로 책에서 많이 쓰는데 낯설 수 있다
# 우리는계속 model = 로 써왔다 -> 친근해졌음 ㅋㅋ

'''
이렇게 하면 10개 node를 가진 hidden layer가 생김 -> 딥러닝모델
model = Sequential()
model.add(Dense(10, activation = 'sigmoid', input_dim = 2))
#model.add(Dense(1)) 
model.summary()
'''
'''
딥러닝 모델이 아니다, input 다음 바로 output
model = Sequential()
model.add(Dense(1, activation = 'sigmoid', input_dim = 2))
model.summary()
'''
# 딥러닝 모델을 짜서 acc를 올려보자
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 2))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid')) # output : 0 or 1
model.summary()



# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# 머신러닝은 compile 과정이 없다
# binary_crossentropy!,
model.fit(x, y, epochs=100, batch_size=1, verbose = 1)  

# 4. 평가 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
x_test = np.array(x_test)

# y_pred = model.predict(x_test)

loss_acc = model.evaluate(x, y) 

# print(x_test, "의 예측 결과 : ", y_pred)
print("loss, acc : ", loss_acc)




