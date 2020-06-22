import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBClassifier(n_estimators = 300, learning_rate = 0.1)

model.fit(x_train, y_train, verbose = True, eval_metric = ['mlogloss', 'merror'],
                            eval_set = [(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds = 10)

results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("ACC : %.2f%%" %(acc * 100.0))



# 시각화
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label = 'Train')

ax.legend()
plt.ylabel('MLog Loss')
plt.title('XGBoost MLog Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, results['validation_1']['merror'], label = 'Test')

ax.legend()
plt.ylabel('MERROR')
plt.title('XGBoost MERROR')

plt.show()
