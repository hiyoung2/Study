import pandas as pd
from sklearn.model_selection import train_test_split, KFold 
from sklearn.model_selection import cross_val_score, GridSearchCV #CV : cross validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# grid, 격자, 그물 모양 / 그물을 던지면 고기를 싹쓸이해서 잡을 수 있다
# grid search : 내가 넣어 놓은 모든 조건을 싹쓸이 해서 모델 실행해준다

# 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv', header = 0)
x = iris.iloc[:, 0:4 ]
y = iris.iloc[:, 4] 
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 44)

parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
    {"C" : [1, 10, 100, 1000], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]},
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.0001]}
]
# SVC에서 제공해주는 파라미터들이다, 내용은 서로 모르니까 퉁 치고 넘어가자
# C는 에포?
# kernerl은 activation 느낌?
# gamma ? 명확히는 모르겠고, 이런 파라미터들이 있다
# C = 1 넣고 kernel을 linear, C = 10 넣고 kernel을 linear,,,

# C = 1 넣고 kernel을 rbf, gemma 0.001, C = 1 넣고, kernel을 rbf, gamma를 0.0001,,,
# C = 10 넣고 kernel을 rbf, gemma 0.001, C = 10 넣고, kernel을 rbf, gamma를 0.0001,,,
# C = 100 넣고 kernel을 rbf, gemma 0.001, C = 100 넣고, kernel을 rbf, gamma를 0.0001,,,
# C = 1000 넣고 kernel을 rbf, gemma 0.001, C = 1000 넣고, kernel을 rbf, gamma를 0.0001,,,

# C = 1 넣고 kernel을 sigmoid, gemma 0.001, C = 1 넣고, kernel을 sigmoid, gamma를 0.0001,,,
# C = 10 넣고 kernel을 sigmoid, gemma 0.001, C = 10 넣고, kernel을 sigmoid, gamma를 0.0001,,,
# C = 100 넣고 kernel을 sigmoid, gemma 0.001, C = 100 넣고, kernel을 sigmoid, gamma를 0.0001,,,
# C = 1000 넣고 kernel을 sigmoid, gemma 0.001, C = 1000 넣고, kernel을 sigmoid, gamma를 0.0001,,,

# parameters 에 넣어 둔 모든 조건들을 실행하고
# 가장 최적의 조건이 무엇이었는지  'model.best_estimator'를 통해 볼 수 있다
# 'mode_best_estimator'의 결과를 보고 어떤 매개변수가 최적이었는지 보면서 수정해가면서 accuracy_score를 높여 볼 수 있다

# train에서만 cv가 일어난다?
# train_test_split을 한 다음에 kfold를 썼으니까

kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(SVC(), parameters, cv = kfold)

# model = GridSearchCV(찐모델, 그 모델의 파라미터, 얼만큼 쪼갤 것인가(여기에는 cv = 5로 해도 똑같음)_
# model에 GridSearchCV를 적용시키겠다!
# () 안에 사용할 모델과 GRID SEARCH에 사용하기 위해 만들어 놓은 파라미터 조합들을 사용할 것이다
# -> 변수 parameters를 적어준다
# 현재 kfold -> n_splits 5로 설정해 둠 / kfold 조건으로 cv(cross validation) 실행

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 : = ", accuracy_score(y_test, y_pred))


# 출력 화면
'''
최적의 매개변수 :  SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
최종 정답률 : =  1.0
'''