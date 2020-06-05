import pandas as pd
from sklearn.model_selection import train_test_split, KFold 
from sklearn.model_selection import cross_val_score, GridSearchCV #CV : cross validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# grid, 격자, 그물 모양 / 그물을 던지면 고기를 싹쓸이해서 잡을 수 있다
# grid search : 내가 넣어 놓은 모든 조건을 싹쓸이 해서 모델 실행해준다
# 레이어 구성할 때 dropout로 랜덤하게 노드를 연산에서 일부 빼줬더니 성능이 더 좋았다
# 드랍아웃과 비슷한 역할을 하는 것이 RandomizedSearchCV이다
# 그리드 서치 모든 조건 넣는다고 성능 무조건 향상? 놉
# 그 중에 일부만 사용 : 랜덤 서치
# 

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
# SVC에서 제공해주는 파라미터들이다, 내용은 서로 모르니까 퉁 치고 넘어가자는 말씀,,,
# C는 에포?
# kernerl은 activation 느낌?
# gemma ? 명확히는 모르겠고, 이런 파라미터들이 있다
# C = 1 넣고 kernel을 linear, C = 10 넣고 kernel을 linear,,,
# C = 1 넣고 kernel을 rbf, gemma 0.001, C = 1 넣고, kernel을 rbf, gemma르르 0.001,,,,

# train에서만 cv가 일어난다?
# train_test_split을 한 다음에 kfold를 썼으니까

kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(SVC(), parameters, cv = kfold)

# model = GridSearchCV(찐모델, 그 모델의 파라미터, 얼만큼 쪼갤 것인가(여기에는 cv = 5로 해도 똑같음)_
# model에 GridSearchCV를 적용시키겠다!
# () 안에 사용할 모델과 GRID SEARCH에 사용하기 위해 만들어 놓은 파라미터 조합들이 있는 
# 변수 PARAMETERS를 적어준다
# 현재 kfold -> n_splits 5로 설정해 둠

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 : = ", accuracy_score(y_test, y_pred))
