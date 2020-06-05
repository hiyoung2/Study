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

# grid, 격자, 그물 모양 / 그물을 던지면 고기를 싹쓸이해서 잡을 수 있다
# grid search : 내가 넣어 놓은 모든 조건을 싹쓸이 해서 모델 실행해준다
# 레이어 구성할 때 dropout로 랜덤하게 노드를 연산에서 일부 빼줬더니 성능이 더 좋았다
# 드랍아웃과 비슷한 역할을 하는 것이 RandomizedSearchCV이다
# 그리드 서치 모든 조건 넣는다고 성능 무조건 향상? 놉
# 그리드 서치 그 중에 일부만 사용 : 랜덤 서치

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


# n_jobs [-1] : 모든 코어 다 사용

kfold = KFold(n_splits = 5, shuffle = True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold, random_state = 2)

# model = GridSearchCV(찐모델, 그 모델의 파라미터, 얼만큼 쪼갤 것인가(여기에는 cv = 5로 해도 똑같음)_
# model에 GridSearchCV를 적용시키겠다!
# () 안에 사용할 모델과 GRID SEARCH에 사용하기 위해 만들어 놓은 파라미터 조합들이 있는 
# 변수 PARAMETERS를 적어준다
# 현재 kfold -> n_splits 5로 설정해 둠

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 : = ", accuracy_score(y_test, y_pred))
