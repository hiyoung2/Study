import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

# model = LGBMClassifier(metric = ["multi_logloss", "multi_error"], objective =  "multiclass")
model = LGBMClassifier(objective = "multiclass")

# objective의 default는 none -> 모델마다 각 설정을 해 줘야 한다?


# 이진분류에서,,,
# 원래는 fit 과정에서 metric에 mlogloss, merror 이런 식으로 넣어주면 되었는데
# LGBM 모델에서는 파라미터로 metric = ["multi_logloss", "multi_error"], objective = "multiclass"를 넣어줘야 한다

# 처음에는 아이리스 데이터인데 boston을 그대로 넣어버렸음
# 근데 돌아는 감(분류든 회귀든 일단 돌아는 감) -> 돌아는 가도 그게 정확하지가 않은 거니까
# 다른 사람들은 에러가 발생하는데 나만 안 나는 게 이상 -> 그럼 다시 살펴봤어야 함
# 파라미터 찾아보는데 열중하다가 지적으로 알게 된 데이터셋 설정 자체의 오류
# 복붙의 폐해
# 꼼꼼하게 하나하나 살펴봐야 한다

model.fit(x_train, y_train, verbose = True, eval_metric = ["multi_logloss", "multi_error"],
                            eval_set = [(x_train, y_train), (x_test, y_test)]
                            )

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("ACC : %.2f%%" %(acc * 100.0)) 

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds :

    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier()

    selection_model.fit(select_x_train, y_train, eval_metric = ["multi_logloss", "multi_error"],
                                             eval_set = [(select_x_train, y_train), (select_x_test, y_test)])
    y_pred = selection_model.predict(select_x_test)

    score = accuracy_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, ACC : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

    pickle.dump(model, open("./model/LGBM/iris/%.4f_pickle.dat" %(score), "wb"))
