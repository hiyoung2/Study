#################################################
#### xor 문제를 해결 할 수 있는 방법을 찾아보자 ####

# 1번 그냥 SVC 모델 쓰기

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0] # xor

# 2. 모델
model = SVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict) # evlauate 친구 / 그냥 score라는 애도 있음(나중에 배움)
print("x_test의 예측 결과 : ", y_predict)

print("acc = ", acc) # acc = 1.0


# 2. 구글링으로 찾은 방법
# from sklearn import svm # svm : support vecotr machine
# from sklearn.metrics import accuracy_score

# # 1. 데이터
# x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
# y_data = [0, 1, 1, 0] # xor

# # 2. 모델
# clf = svm.SVC()

# # 3. 훈련
# clf.fit(x_data, y_data)

# # 4. 평가, 예측
# y_predict = clf.predict(x_data)
# print(y_predict) # [0 1 1 0]

# acc = accuracy_score([0, 1, 1, 0], y_predict)
# print("acc = ", acc) # acc =  1.0

# 3. MLP로 구성 (2번 접어 두 선이 생김?)