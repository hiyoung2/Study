import numpy as np

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

# model = XGBClassifier(n_estimators = 300, learning_rate = 0.1)
# model.fit(x_train, y_train, verbose = True, eval_metric = ['logloss', 'error'],
                            # eval_set = [(x_train, y_train), (x_test, y_test)],
                            # early_stopping_rounds = 10)
model = XGBClassifier()
model.fit(x_train, y_train)




results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("ACC :", acc)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds :

    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)


    selection_model = XGBClassifier(n_estimators = 300, cv = 5, n_jobs = -1)
    
    selection_model.fit(select_x_train, y_train, verbose = True, eval_metric=["logloss", "error"],
                                                 eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                                                 early_stopping_rounds=10)

    y_pred = selection_model.predict(select_x_test)

    results = selection_model.evals_result()
    print("eval's results :", results)

    score = accuracy_score(y_test, y_pred)

    print("Trhesh = %.3f, n = %d, ACC : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


