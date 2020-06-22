'''
SelectFromModel에
1. 회귀        m29_eval1
2. 이진 분류    m29_eval2
3. 다중 분류    m29_eval3

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용
3. plot으로 그릴 것

4. 결과는 주석으로 소스 하단에 표시
'''

# 그림을 그려 봅시다!
# 기본 코드로만 구성(나머지는 알아서 찾아보기)

import numpy as np
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBRegressor(n_estimators =1300, learning_rate = 0.1)
# n_estimators default = 100

model.fit(x_train, y_train, verbose = True, eval_metric = ["logloss", "rmse"],
                            eval_set = [(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds = 10)

# results = model.evals_result()
# print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("R2 : %.2f%%" %(r2 * 100.0)) 

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds :

    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    "colsample_bylevel" : [0.6, 0.7, 0.9]}
    ]



    selection_model = XGBRegressor(n_estimators = 100, learning_rate = 0.09, max_depth = 4, cv = 5, n_jobs = -1)

    selection_model.fit(select_x_train, y_train, eval_metric = ["logloss", "rmse"], 
                                                eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                                                early_stopping_rounds = 5)

    y_pred = selection_model.predict(select_x_test)

    results = selection_model.evals_result()
    print("eval's results :", results)

    score = r2_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

    selection_model.save_model("./model/SFM/boston/%.4f_save.dat"%(score))






