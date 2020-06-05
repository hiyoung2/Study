import pandas as pd
from sklearn.model_selection import train_test_split, KFold , cross_val_score
# KFold, cross_val_across import11

from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators 

import warnings 

warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4 ]
y = iris.iloc[:, 4] 


print(x)
print(y)

'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44
)
'''
# KFold 사용하기
kfold = KFold(n_splits = 5, shuffle = True)
# 데이터를 kfold로 5개씩 나누겠다
# 한 작업당 20%씩 쓰게 됨 / 한 작업 단위 데이터는 80% 훈련, 20% 테스트(검증)

allAlgorithms = all_estimators(type_filter = 'classifier') 


for (name, algorithm) in allAlgorithms : 
    model = algorithm()

    scores = cross_val_score(model, x, y, cv = kfold) 
    # model.fit 대신에 위를 사용한다
    # 우리는 이 모델을 분리하지 않은 x, y 를 넣으면 kfold가 적용되어 총 데이터가 5개를 잘라서 계속 score를 내 주겠다!
    # 여기서 score는 acc!
    # 딥러닝에서도 사용할 수 있다ㄴ

    print(name, "의 정답률 = ")
    print(scores)


# 모델별로 5개씩의 scores, 즉 acc가 나온다


'''
AdaBoostClassifier 의 정답률 = 
[0.96666667 0.9        0.96666667 0.93333333 0.96666667]
BaggingClassifier 의 정답률 = 
[0.93333333 1.         0.93333333 0.96666667 0.93333333]
BernoulliNB 의 정답률 = 
[0.3        0.23333333 0.13333333 0.26666667 0.3       ]
CalibratedClassifierCV 의 정답률 = 
[0.96666667 0.96666667 0.96666667 0.93333333 0.86666667]
ComplementNB 의 정답률 = 
[0.73333333 0.56666667 0.7        0.63333333 0.7       ]
DecisionTreeClassifier 의 정답률 = 
[1.         0.96666667 0.93333333 0.96666667 0.93333333]
ExtraTreeClassifier 의 정답률 = 
[0.86666667 0.93333333 0.96666667 0.9        0.96666667]
ExtraTreesClassifier 의 정답률 = 
[0.93333333 0.86666667 0.93333333 0.96666667 1.        ]
GaussianNB 의 정답률 = 
[0.93333333 0.96666667 0.96666667 0.93333333 0.96666667]
GaussianProcessClassifier 의 정답률 = 
[0.93333333 0.96666667 0.96666667 1.         0.96666667]
GradientBoostingClassifier 의 정답률 = 
[0.96666667 0.96666667 0.93333333 1.         0.93333333]
KNeighborsClassifier 의 정답률 =
[0.96666667 0.9        0.96666667 1.         1.        ]
LabelPropagation 의 정답률 =
[0.96666667 0.96666667 1.         0.96666667 0.93333333]
LabelSpreading 의 정답률 =
[0.93333333 1.         0.93333333 1.         0.93333333]
LinearDiscriminantAnalysis 의 정답률 =
[0.93333333 1.         1.         1.         0.96666667]
LinearSVC 의 정답률 =
[0.93333333 1.         1.         0.9        0.96666667]
LogisticRegression 의 정답률 =
[0.96666667 0.96666667 0.96666667 0.9        0.93333333]
LogisticRegressionCV 의 정답률 =
[0.93333333 0.83333333 0.96666667 1.         1.        ]
MLPClassifier 의 정답률 =
[0.96666667 0.96666667 0.96666667 0.96666667 1.        ]
MultinomialNB 의 정답률 =
[0.93333333 1.         0.56666667 0.9        0.86666667]
NearestCentroid 의 정답률 =
[0.93333333 0.96666667 0.93333333 0.9        0.83333333]
NuSVC 의 정답률 =
[1.         0.96666667 0.96666667 0.93333333 0.96666667]
PassiveAggressiveClassifier 의 정답률 =
[0.7        1.         0.8        0.7        0.66666667]
Perceptron 의 정답률 =
[0.9        0.56666667 0.8        0.63333333 0.6       ]
QuadraticDiscriminantAnalysis 의 정답률 =
[0.96666667 1.         0.96666667 0.96666667 0.96666667]
RadiusNeighborsClassifier 의 정답률 =
[0.96666667 0.93333333 0.96666667 0.96666667 0.9       ]
RandomForestClassifier 의 정답률 =
[0.96666667 0.93333333 1.         1.         0.93333333]
RidgeClassifier 의 정답률 =
[0.76666667 0.9        0.7        0.83333333 0.9       ]
RidgeClassifierCV 의 정답률 =
[0.76666667 0.7        0.86666667 0.96666667 0.76666667]
SGDClassifier 의 정답률 =
[0.66666667 0.43333333 0.83333333 0.43333333 0.5       ]
SVC 의 정답률 =
[1.         1.         0.96666667 0.96666667 0.96666667]
'''
   


import sklearn
print(sklearn.__version__)


