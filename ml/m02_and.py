# 머신러닝의 기본

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 0, 0, 1]

# 0, 0 -> 0, 1, 0 -> 0, 0, 1 -> 0, 1, 1 -> 1 : and!(둘 다 참이어야 참)

# 2. 모델
model = LinearSVC() # linear = 선형 -> 회귀모델이다 / linear, regression : 회귀
                    # calssify = 분류모델
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

acc = accuracy_score([0, 0, 0, 1], y_predict) # predict도 케라스와 같은데 accuracy_score만 다르다
                                              # evaluate 대신 score라는 걸 쓴다
print("x_test의 예측 결과 : ", y_predict)

print("acc = ", acc)