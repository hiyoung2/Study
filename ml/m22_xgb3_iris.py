# 과적합 방지
# 1. 훈련데이터량을 늘린다(훈련 많이 하는 만큼 방지된다)
# 2. 피처수를 줄인다
# 3. regularizaton

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 회귀 모델
dataset = load_iris()
x = dataset.data
y = dataset.target

print("x.shape :", x.shape) # (506, 13)
print("y.shape :", y.shape) # (506, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


# tree(나무) 모델 여럿 -> ensemble의 randomforest(숲)
# 여기서 업그레이드 된 게 boost 계열 모델
# tree 게열에서 이어지기 때문에 파라미터는 바뀐 것들 있지만
# 트리 구조의 특성과 거의 동일
# 트리 구조 : 전처리를 안 해도 된다, 결측치 제거 안 해줘도 된다
# xgboost : 속도가 빠르다(딥러닝에 비해), 
#           일반 머신러닝보다는 조금 느림(앙상블모델이니까, 여러 나무를 쓰므로)
# 결측치 nan 보간법 안 해줘도 되는데, 할 수는 있다
# 자동으로 했을 땐 평타 85이니까 선택사항이다
# xgboost kaggle에서 5년동안 우승 모델이었다
# 지금은 케라스로 바뀌었지만

# xgboost에서 반드시 알고 있어야 할 파라미터
# 요 네가지만 알고 있으면 된다(당연한 것, xgboost 모델)
n_estimators = 100 # 추정치, 나무의 숫자
learning_rate = 0.09 # 학습률(디폴트 : 0.01), 튜닝 핵심키워드 중 하나(노드, 레이어 등 건드리는 것들보다)
colsample_bytree = 0.7 # 트리의 샘플을 얼마나 할 것인가 (1 = 100%, 디폴트), 통상적으로 0.6 ~ 0.9 사이를 많이 쓴다
colsample_bylevel = 0.7 # max = 1, 0.6 ~ 0.9 사이 많이 쓴다 (cv 하면 0.6, 0.7, 0.8, 0.9)

max_depth = 7
n_jobs = -1 # 딥러닝 아닐 경우에는 무조건 n_jobs = -1을 쓰자(딥러닝은 터짐)


# cv, feature_importance 이 두 개 꼭 해야 함

model = XGBClassifier(max_dapthe = max_depth, learning_rate = learning_rate,
                      n_estimators = n_estimators, n_jobs = n_jobs,
                      colsample_bytree = colsample_bytree)

                    #   colsample_bylevel = colsample_bylevel)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("점수 :", score)

print(model.feature_importances_)

plot_importance(model)
# plt.show()