# 과적합 방지
# 1. 훈련데이터량을 늘린다(훈련 많이 하는 만큼 방지된다)
# 2. 피처수를 줄인다
# 3. regularizaton

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt

# 회귀 모델
# dataset = load_boston()
dataset = load_boston()
x = dataset['data']
y = dataset['target']


print("x.shape :", x.shape) # (506, 13)
print("y.shape :", y.shape) # (506, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


n_estimators = 100 
learning_rate = 0.09 
colsample_bytree = 0.7 
colsample_bylevel = 0.7 

max_depth = 7
# n_jobs = -1 


parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate" :[0.1, 0.3, 0.5, 0.01],
     "max_depth" : [4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01],
     "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate" : [0.1, 0.001, 0.5],
     "max_depth" : [4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
     "colsample_bylevel" : [0.6, 0.7, 0.9]}
]



model = GridSearchCV(XGBRegressor(), parameters, cv = 5, n_jobs = -1)
model.fit(x_train, y_train)

print("=============================================")
print(model.best_estimator_)
print("=============================================")
print(model.best_params_)
print("=============================================")

score = model.score(x_test, y_test)
print("점수 :", score)

# print(model.feature_importances_)

# plot_importance(model)
# plt.show()


