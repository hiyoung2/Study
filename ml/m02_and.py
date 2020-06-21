# 머신러닝의 기본

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 0, 0, 1]

# 0, 0 -> 0, 1, 0 -> 0, 0, 1 -> 0, 1, 1 -> 1 : and!(둘 다 참이어야 참)

# 2. 모델
model = LinearSVC() 

# 보통, 모델 이름이
# linear = 선형 -> 회귀모델이다 / linear, regression : 회귀
# calssifier : 분류모델
# SVC에 대해 혼자 알아보자,,
# 케라스에서 구조를 익혀서 머신러닝을 쉽게 받아들일 수 있다
# 머신러닝은 모델 정말 간단하다
# 머신러닝은 모델이 이미 만들어져있다
# customize 할 것은 ()안에 뭘 넣을지 정하는 것밖에 없다

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 0, 0, 1], y_predict) 
# predict도 케라스와 같은데 accuracy_score만 다르다
# evaluate 대신 score라는 걸 쓴다

print("x_test의 예측 결과 : ", y_predict)
print("acc = ", acc)


# linearSVC는 어디서 가져오는가? -> SVM 모듈에서
# Support Vector Machine, 서포트벡터머신은 
# 매우 강력하고 선형이나 비선형 분류, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델
# SVM은 특히 복잡한 분류 문제에 잘 들어맞으며 작거나 중간 크기의 데이터셋에 적합하다
