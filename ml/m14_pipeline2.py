# RandomizedSearchCV + Pipeline 

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터

iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 43
)

# GridS/ RandomS를 사용할 때는 앞에 우리가 정한 모델명을 적어줘야 한다
# 파라미터 이름 앞에 모델명 + 언더바 2개를 써야 GrideSearchCV(+RandomizedGridSearchCV)가 알아 듣고 실행함
# for GridSearchCV(or RandomizedSearchCV) with Pipeline
parameters = [
    {"svm__C":[1, 10, 100, 1000], "svm__kernel":["linear"], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 10, 100], "svm__kernel":["rbf"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    {"svm__C":[1, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma" : [0.001, 0.0001], "svm__degree" : [3, 6, 9]},
    
]

# '__' underbar 2개 빼고 하니까
# ValueError: Invalid parameter kernel for estimator Pipeline 에러 메세지 발생함
# '__'를 써 주는 건 문법적인 거라고 보면 된다

# Grid Search만 할 때는 이렇게 써도 됨(pipeline 사용 안 할 시)
# pipeline에선 안 먹힌다

# for GridSearchCV(or RandomizedSearchCV), not Pipeline
# parameters = [
#     {"C":[1, 10, 100, 1000], "kernel":["linear"]},
#     {"C":[1, 10, 100, 1000], "kernel":["rbf"], "gamma" : [0.001, 0.0001]},
#     {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma" : [0.001, 0.0001]}
# ]


# for GridSearhCV(or RandomizedSearchCV) with make_pipeline parameters
# parameters = [
#     {"svc__C":[1, 10, 100, 1000], "svc__kernel":["linear"]},
#     {"svc__C":[1, 10, 100, 1000], "svc__kernel":["rbf"], "svc__gamma" : [0.001, 0.0001]},
#     {"svc__C":[1, 10, 100, 1000], "svc__kernel":["sigmoid"], "svc__gamma" : [0.001, 0.0001]}
# ]


# 2. 모델 

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
# 기존 Pipeline은 우리가 전처리명, 모델명 이름을 'scaler', 'svm' 이렇게 정했음

# svm -> svc로 바꿔 봄
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svc', SVC())])
# 에러 발생
# 파이프라인 모델 이름 명시한 것은 파라미터 앞에 붙이는 것과 동일하게 해 줘야 한다
# 'svc'로 바꿨으면 "svc__C" 이런 식으로 해 줘야 함


# maek_pipeline
pipe = make_pipeline(MinMaxScaler(), SVC())
# 기존 파라미터로 함께 실행하면 에러가 뜬다
# make_pipeline은 모델명을 우리가 따로 명시하지 않음
# 그러면 실제 사용하는 모델명을 (대,소문자 상관없음) ex) svc__C로 해 주면 된다
# 위 for make_pipeline parameters 참조


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



