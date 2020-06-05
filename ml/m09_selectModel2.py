# 보스턴 모델링 하시오.
# m09_selectModel.py 복붙 -> 회귀모델에 맞게끔 수정

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators 
# all_estimators 안에 머신러닝 모델들이 다 들어 있다

import warnings # 워닝 에러 그냥 넘김

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv', header = 1)

x = boston.iloc[:, 0:13 ]
y = boston.iloc[:, 13] 

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44, shuffle = True
)

allAlgorithms = all_estimators(type_filter = 'regressor') 


for (name, algorithm) in allAlgorithms : 
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 : ", r2_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)

model.fit(x_train, y_train)
r2 = model.score(x_test, y_test)

print("r2score : ", r2_score(y_test, y_pred))
print("r2       : ", r2)


'''
ARDRegression 의 정답률 :  0.7512651671065367
AdaBoostRegressor 의 정답률 :  0.8368491783672969
BaggingRegressor 의 정답률 :  0.8888467103738054
BayesianRidge 의 정답률 :  0.7444777786443536
CCA 의 정답률 :  0.7270542664211515
DecisionTreeRegressor 의 정답률 :  0.821567473382704
ElasticNet 의 정답률 :  0.699050089875551
ElasticNetCV 의 정답률 :  0.6902681369495265
ExtraTreeRegressor 의 정답률 :  0.7834115209458854
ExtraTreesRegressor 의 정답률 :  0.8968238625556036
GaussianProcessRegressor 의 정답률 :  -5.639147690233129
GradientBoostingRegressor 의 정답률 :  0.8955186072692273
HuberRegressor 의 정답률 :  0.7060665322978028
KNeighborsRegressor 의 정답률 :  0.6390759816821279
KernelRidge 의 정답률 :  0.7744886782293467
Lars 의 정답률 :  0.7521800808693164
LarsCV 의 정답률 :  0.7521800808693164
Lasso 의 정답률 :  0.6855879495660049
LassoCV 의 정답률 :  0.71540574604873
LassoLars 의 정답률 :  -0.0007982049217318821
LassoLarsCV 의 정답률 :  0.7521800808693164
LassoLarsIC 의 정답률 :  0.754094595988446
LinearRegression 의 정답률 :  0.7521800808693132
LinearSVR 의 정답률 :  0.45049343144215637
MLPRegressor 의 정답률 :  0.5434029988911482
'''