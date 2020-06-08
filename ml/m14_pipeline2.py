# RandomizedSearchCV + Pipeline 

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# 1. 데이터

iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 43
)

# GridS/ RandomS 사용할 매개 변수
parameters = [
    {"svm__C":[1, 10, 100, 1000], "svm__kernel":["linear"], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 10, 100], "svm__kernel":["rbf"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    
]

# '__' underbar 2개 빼고 하니까
# ValueError: Invalid parameter kernel for estimator Pipeline 에러 메세지 발생함
# '__'를 써 주는 건 문법적인 거라고 보면 된다
# parameters = [
#     {"C":[1, 10, 100, 1000], "kernel":["linear"]},
#     {"C":[1, 10, 100, 1000], "kernel":["rbf"], "gamma" : [0.001, 0.0001]},
#     {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma" : [0.001, 0.0001]}
# ]

# 2. 모델 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

# pipe = make_pipeline(MinMaxScaler(), SVC())
# make_pipline을 버려라ㅋㅋㅋㅋㅋㅋㅋㅋ

model = RandomizedSearchCV(pipe, parameters, cv = 5)

# 3. 훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)
'''
최적의 매개변수 : Pipeline(memory=None,
         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('svm',
                 SVC(C=1000, break_ties=False, cache_size=200,
                     class_weight=None, coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma=0.001,
                     kernel='sigmoid', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=0.001,
                     verbose=False))],
         verbose=False)
acc : 0.9333333333333333
'''

# print("최적의 매개변수 :", model.best_params_)
'''
최적의 매개변수 : {'svm__kernel': 'sigmoid', 'svm__gamma': 0.001, 'svm__C': 1000}
acc : 0.9333333333333333
'''

print("acc :", acc)

import sklearn as sk
print("sklearn :", sk.__version__)



