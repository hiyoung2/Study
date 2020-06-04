# # 머신러닝의 기본

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0] # xor

# 2. 모델
model = LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict)
print("x_test의 예측 결과 : ", y_predict)

print("acc = ", acc) # acc = 0.5 

# xor을 해결해보자
# m03_xor2, m03_xor3 파일로
