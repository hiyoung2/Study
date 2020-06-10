# Decision Tree, 결정 트리 # 차차차, 렛잇고 차차차

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x = cancer['data']
y = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 42, shuffle = True
)

model = DecisionTreeClassifier(max_depth = 4)
# 보통 3이나 4으로 지정
# 5 이상이면 과적합 발생 확률 아주 높음
#

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print("acc :" , acc)

print(model.feature_importances_)
'''
30개가 출력
breast_cancer : column 30
[0.         0.05959094 0.         0.         0.         0.00639525
 0.         0.70458252 0.         0.         0.         0.
 0.         0.01221069 0.         0.         0.0189077  0.0162341
 0.         0.         0.05329492 0.         0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]
 8번째 컬럼(0.70458252) -> acc에 가장 큰 영향을 끼치는 놈이다
 0인 애들은 필요 없다고 보면 된다
 없애 주면 10개 정도의 column이 남음
 가장 높은 놈들 5개만 추려도 좋다
 PCA와 비슷하다
'''

# Decision Tree로 모델 짜는 경우는 없다
# feature importance를 보기 위한 도구일 뿐
# 나무들마다도 feature importance가 있는데
# 숲에도 있을 것
# 숲에서는 feature importance가 다르게 나올 수 있다
# 숲과 나무를 비교해서 우리가 선택해야 한다

# tree 구조의 장점(ree, randomforest, xgbooster 모두)
# 전처리가 필요 없다
# 단점
# 과적합이 잘 된다(띠용)