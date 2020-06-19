from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

dataset = load_boston()
x = dataset.data
y = dataset.target

# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

n_jobs = -1

    
parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01, 0.09],
    "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01, 0.09],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.7, 0.9, 1],
    "colsample_bylevel" : [0.6, 0.7, 0.9]}
    ]

model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = n_jobs)


model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R2 :", score)

print("========================================")
print(model.best_estimator_)
print("========================================")
print(model.best_estimator_.feature_importances_)

# feature engineering
print("========================================")
thresholds = np.sort(model.best_estimator_)
print(thresholds)




for thresh in thresholds: # 컬럼 수만큼 돈다! 빙글 빙글
               
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)

    print("Trersh=%.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))




# 과제
# 그리드 서치까지 엮어라
# 파라미터 : median, threshold 정리

# 데이콘 적용해라 71개 컬럼

# 월요일까지 적용한 데이콘 소스, 데이콘 성적 메일로 보내라

# 메일 제목 : 하인영 24등 
