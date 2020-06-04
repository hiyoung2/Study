#################################################
#### xor 문제를 해결 할 수 있는 방법을 찾아보자 ####

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0] # xor

# 2. 모델
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1) # 각 개체를 한 개씩만 연결하겠다

# 최근접 이웃, 누굴 근접 이웃으로 둘 것인가, 매개변수를 둬야 한다
# n_neighbors

'''
0 1 0의 이웃 1, 1의 이웃 0
1 0
'''

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict) 
print("x_test의 예측 결과 : ", y_predict)

print("acc = ", acc) 

# n_neighbors = 1로 하니까 acc = 1.0
# n_neighbors = 2로 하니까 acc = 0.5
# 데이터가 4개밖에 안 되는데 2개씩 이웃으로 붙여버리니까 acc가 떨어짐

