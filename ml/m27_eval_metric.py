from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target
# 다음과 같이 사용할 수도 있음
# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.1)

# 딥러닝 훈련과정 볼 수 있게 하는 것 : verbose, 머신러닝에도 있었다
# eval_metrics : 
# 딥러닝에서 metrics 가 있었음 , metrics : acc 하면 acc를 보여줬음
# metrics에 들어가는 것들 : rmse, mae, (회귀 지표) logloss, (loss), error,(acc의 반대 개념) auc(acc의 친구) 가 있다
# error 가 0.2라면 acc는 0.8이라고 볼 수 있다
# 현재는 error 정도까지만 알고 있으면 된다

model.fit(x_train, y_train, verbose = True, eval_metric = ["logloss", "rmse"],
                            eval_set = [(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds = 10)

# xgb 에서는 score 뿐 아니라 evals_result로 평가 결과를 볼 수 있다
results = model.evals_result()
# print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
# y_pred, y_test 이렇게 순서 바꿔 넣어도 상관 없다

print("R2 : %.2f%%" %(r2 * 100.0)) # R2 : 93.29%


'''
Stopping. Best iteration:
[48]    validation_0-logloss:-791.72449 validation_0-rmse:0.96663       validation_1-logloss:-799.52972 validation_1-rmse:2.37591

R2 : 93.25%
'''