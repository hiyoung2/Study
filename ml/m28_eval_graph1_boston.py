# 그림을 그려 봅시다!
# 기본 코드로만 구성(나머지는 알아서 찾아보기)

from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBRegressor(n_estimators = 300, learning_rate = 0.1)

# 딥러닝에서 metrics 가 있었음 , metrics : acc 하면 acc를 보여줬음
# metrics에 들어가는 것들 : rmse, mae, (회귀 지표) logloss, (loss), error,(acc의 반대 개념) auc(acc의 친구) 가 있다

model.fit(x_train, y_train, verbose = True, eval_metric = ["logloss", "rmse"],
                            eval_set = [(x_train, y_train), (x_test, y_test)])
                            # early_stopping_rounds = 100)

results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("R2 : %.2f%%" %(r2 * 100.0)) # R2 : 93.29%

# 그래프 그리기
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['logloss'], label = 'Test')

ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, results['validation_1']['rmse'], label = 'Test')

ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')

plt.show()