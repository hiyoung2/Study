import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators 
# all_estimators 안에 머신러닝 모델들이 다 들어 있다

import warnings # 워닝 에러 그냥 넘김

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4 ] # 0, 1, 2, 3 x의 column
y = iris.iloc[:, 4] # 4번째 column이 y

#pandas iloc, loc 알고 있어야 한다

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44
)

allAlgorithms = all_estimators(type_filter = 'classifier') 

# type_filter = 'classifier' 분류 모델들을 다 돌릴 수 있음
# 옵션은 디폴트로 돌아간 것
# type_filter 에 회귀를 넣어주면 회귀 모델들을 다 돌릴 수 있겠지

for (name, algorithm) in allAlgorithms : 
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 : ", accuracy_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print("acc_score : ", accuracy_score(y_test, y_pred))
print("acc       : ", acc)
# acc_score :  0.9666666666666667
# acc       :  0.9666666666666667


# 26개의 모델을 한 번에 돌림! wow 정말 빠르다
# sklearn 0.20 모델에서 제공하는 모든 모델이다
'''
AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9666666666666667
ComplementNB 의 정답률 :  0.7
DecisionTreeClassifier 의 정답률 :  0.8666666666666667
ExtraTreeClassifier 의 정답률 :  0.9333333333333333
ExtraTreesClassifier 의 정답률 :  0.9
GaussianNB 의 정답률 :  0.9333333333333333
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9333333333333333
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9666666666666667
LogisticRegression 의 정답률 :  1.0

# LogisticRegrssion 은 분류다!!!!!!!!!!!!!!

LogisticRegressionCV 의 정답률 :  0.9
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  0.8666666666666667
NearestCentroid 의 정답률 :  0.9
NuSVC 의 정답률 :  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
Perceptron 의 정답률 :  0.5333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.8333333333333334
RidgeClassifierCV 의 정답률 :  0.8333333333333334
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  0.9666666666666667
'''