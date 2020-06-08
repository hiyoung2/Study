# m11_gridSearch copy
# breast_cancer 적용

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV #CV : cross validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 레이어 구성할 때 dropout로 랜덤하게 노드를 연산에서 일부 빼줬더니 성능이 더 좋았다
# 드랍아웃과 비슷한 역할을 하는 것이 RandomizedSearchCV이다
# 그리드 서치 모든 조건 넣는다고 성능 무조건 향상? 놉
# 그리드 서치 그 중에 일부만 사용 : 랜덤 서치

# 핸즈온 머신러닝 p118
# GridSearchCV 는 비교적 적은 수의 조합을 탐구할 때 괜찮음
# but, 하이퍼파라미터 탐색 공간이 커지면 RandomizedSearchCV를 사용하는 편이 더 좋다
# RandomizedSearchCV는 GridSearchCV와 거의 같은 방식으로 사용하지만, 가능한 모든 조합을 시도하는 대신
# 각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다


# 케라스에서 그리드 서치 -> 정말 오래 걸림 (일부 학원은 실습 못할 수준)
# 작은 모델들로 어떻게든 돌려는 볼 것

# 파라미터에 집착하지 말고 그리드서치, randomizedSearch 사용방법만 잘 익혀 둬라

# 1. 데이터
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44)

print("x_train.shape : ", x_train.shape) # (455, 30)
print("x_test.shape : ", x_test.shape)   # (114, 30)
print("y_train.shape : ", y_train.shape) # (455,)
print("y_test.shape : ", y_test.shape)   # (114,)

parameters = {
    "n_estimators" : [10, 20, 30, 100], "max_depth" : [4, 8, 10, 12, 20], 
    "min_samples_leaf" : [3, 5, 7, 9], "min_samples_split" : [3, 5, 7, 9],
    "n_jobs" : [-1], "criterion" : ["gini"]}

# parameters = {
#     "n_estimators" : [10], "max_depth" : [10], 
#     "min_samples_leaf" : [10], "min_samples_split" : [10],
#      }

# n_jobs [-1] : 모든 코어 다 사용

kfold = KFold(n_splits = 5, shuffle = True) 
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold, random_state = 2)

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 : = ", accuracy_score(y_test, y_pred))
