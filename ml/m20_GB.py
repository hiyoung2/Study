# Gradient Boost
# randomforest, xgbosst 사이에 나옴
# 19 copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

print("x.shape :", x.shape)


x_train, x_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, train_size = 0.8, random_state = 42, shuffle = True
)

# model = DecisionTreeClassifier(max_depth = 4)
# model = RandomForestClassifier(max_features = "auto", n_estimators = 1000, n_jobs = -1)

model = GradientBoostingClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print("acc :" , acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt

import numpy as np

def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()
