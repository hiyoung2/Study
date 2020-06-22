from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.1)


model.fit(x_train, y_train, verbose = True, eval_metric= "error", eval_set = [(x_train, y_train), (x_test, y_test)])

results = model.evals_result()
# print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("ACC :", acc)


###########################################################################################################
# pickle, joblib 비교 위해 주석
# import pickle # python 에서 제공하는 것은 from 할 것도 없이 import 하면 된다
# pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb")) # write binary = wb


# from joblib import dump, load
# dump(model, "./model/xgb_save/cancer.joblib.dat")
# 동일하게 이렇게도 쓴다
import joblib
joblib.dump(model,"./model/xgb_save/cancer.joblib.dat") # .dat:확장자명(절대적이진 않지만 약속된 것이라 생각)

print("저장완료") # 출력이 되면 위의 모델 저장은 문제가 없는 것


# model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", "rb")) # read의 r
model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")

print("불러오기 완료")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("ACC :", acc)


'''
ACC : 0.9736842105263158
저장완료
불러오기 완료
ACC : 0.9736842105263158
'''