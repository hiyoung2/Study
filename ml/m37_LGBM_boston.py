import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = LGBMClassifier()




model.fit(x_train, y_train, verbose = True, eval_metric = ["logloss", "rmse"],
                            eval_set = [(x_train, y_train), (x_test, y_test)],
                            )

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("R2 : %.2f%%" %(r2 * 100.0)) 

thresholds = np.sort(model.feature_importances_)
print(thresholds)



for thresh in thresholds :

    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier(n_iter = 30, max_depth = -1)

    selection_model.fit(select_x_train, y_train, eval_metric = ["logloss", "error"], 
                                                eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                                                )

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

    pickle.dump(model, open("./model/LGBM/cancer/%.4f_pickle.dat" %(score), "wb"))







