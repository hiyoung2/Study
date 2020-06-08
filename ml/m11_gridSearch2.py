# m11_gridSearch copy
# breast_cancer 적용

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, KFold 
from sklearn.model_selection import cross_val_score, GridSearchCV #CV : cross validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44, shuffle = True)

print("x_train.shape : ", x_train.shape) # (455, 30)
print("x_test.shape : ", x_test.shape)   # (114, 30)
print("y_train.shape : ", y_train.shape) # (455,)
print("y_test.shape : ", y_test.shape)   # (114,)

parameters = [
    {"n_estimators" : [10, 20, 30], "max_depth" : [10, 20, 30], 
    "min_samples_leaf" : [5, 10], "min_samples_split" : [5, 10],
    "n_jobs" : [-1]}
]

# parameters = [
#     {"n_estimators" : [10, 20, 30]}
# ]

# n_jobs [-1] : 모든 코어 다 사용
# 'RandomFroestClassifier 에서 제공해주는 파라미터들 적용(엄청 많음)
# parameter들 입력 안 해도 실행이 된다? -> default 들이 각각 존재
# 위처럼 다른 매개변수 다 지우고 n_estimators만 입력해도 아래와 같이 출력된다
'''
최적의 매개변수 :  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
최종 정답률 : =  0.9649122807017544
'''

kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold)

# model = GridSearchCV(찐모델, 그 모델의 파라미터, 얼만큼 쪼갤 것인가(여기에는 cv = 5로 해도 똑같음)_
# model에 GridSearchCV를 적용시키겠다!
# () 안에 사용할 모델과 GRID SEARCH에 사용하기 위해 만들어 놓은 파라미터 조합들이 있는 
# 변수 PARAMETERS를 적어준다
# 현재 kfold -> n_splits 5로 설정해 둠

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_) # model.best_estimator : 매개변수 조합들 중 가장 결괏값이 좋은 최적의 매개변수를 보여준다
                                                  # 기록해두고 비교할 수 있다
y_pred = model.predict(x_test)
print("최종 정답률 : = ", accuracy_score(y_test, y_pred))
