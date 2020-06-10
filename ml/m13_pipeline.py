# Pipeline

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. 데이터

iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 43
)

# 2. 모델 
# model = SVC() #svc_model = SVC() 이렇게 써도 상관 없다
# pipeline 도 제공될 것 같다 땡겨와 보자
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
#                  전처리방법                  모델명
# target 값(즉, y data)은 알아서 전처리 하지 않는다



# make_pipeline 사용법
pipe = make_pipeline(MinMaxScaler(), SVC())
# 그냥 사용할 전처리방법, 모델명만 써 주면 된다
# make_pipeline 버전문제가 아니라 표기방법이 달랐던 것




pipe.fit(x_train, y_train)

print("acc :", pipe.score(x_test, y_test))


'''
MinMaxScaler 쓰니까
FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
경고메세지가 뜸
acc : 0.9666666666666667


StandardScaler 사용
acc : 1.0
'''





