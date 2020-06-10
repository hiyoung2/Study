# RandomizedSearchCV + Pipeline 

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터

iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 43
)

# for GridSearchCV(or RandomizedSearchCV) with Pipeline
parameters = [
    {"randomforestclassifier__n_estimators":[10, 20, 30], "randomforestclassifier__max_depth":[4, 8, 10],
     "randomforestclassifier__max_features" : ["auto"], "randomforestclassifier__max_leaf_nodes":[2, 4, 8]},
    {"randomforestclassifier__n_estimators":[10, 20, 30], "randomforestclassifier__max_depth":[4, 8, 10], 
     "randomforestclassifier__max_leaf_nodes":[3, 6, 9]},
    {"randomforestclassifier__n_estimators":[10, 20, 30], "randomforestclassifier__max_depth":[3, 5, 7, 9]}
]

# parameters = {
#     "n_estimators" : [10, 20, 30, 100], "max_depth" : [4, 8, 10, 12, 20], 
#     "min_samples_leaf" : [3, 5, 7, 9], "min_samples_split" : [3, 5, 7, 9],
#     "n_jobs" : [-1], "criterion" : ["gini"]}


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

# Pipeline
# pipe = Pipeline([("scaler", MinMaxScaler()), ('ensemble', RandomForestClassifier())])


# maek_pipeline
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())



model = RandomizedSearchCV(pipe, parameters, cv = 5)

# 3. 훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)
# print("최적의 매개변수 :", model.best_params_)


print("acc :", acc)

import sklearn as sk
print("sklearn :", sk.__version__)


'''
최적의 매개변수 : Pipeline(memory=None,
         steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('randomforestclassifier',
                 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                        max_depth=4, max_features='auto',
                                        max_leaf_nodes=8, max_samples=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=10, n_jobs=None,
                                        oob_score=False, random_state=None,
                                        verbose=0, warm_start=False))],
         verbose=False)
acc : 0.9333333333333333
sklearn : 0.22.1
'''