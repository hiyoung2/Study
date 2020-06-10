# feature_importances

# Decision Tree, 결정 트리 # 차차차, 렛잇고 차차차

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

print("x.shape :", x.shape)


x_train, x_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, train_size = 0.8, random_state = 42, shuffle = True
)

model = DecisionTreeClassifier(max_depth = 4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print("acc :" , acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt

import numpy as np

def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()




'''
소스 분석
def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1] # cancer.data = x의 shape 중 [1], 즉 column이 n_features
                                      # (569, 30) 중 [1]에 해당하는 column 갯수 30개를 지칭
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

barh : 가로 막대 그래프를 그리는 함수
(cf. bar : 세로 막대 그래프를 그리는 함수)
- align : default는 center, 다른 option : edge , 정렬방식을 지정하는 것

ticks : tick을 표시할 값, 그리고 해당 위치에 어떤 label을 작성할 지 지정한다
- 지금은 yticks만 사용, data의 column 명을 label로 사용했다
- xticks도 사용 가능함
- 플롯이나 차트에서 축상의 위치 표시 지점을 tick 이라고 한다
- 이 tick에 써진 숫자 혹은 글자를 tick label이라고 한다
- tick의 위치나 label은 Matplotlib가 자동으로 정해주지만
- 만약 수동으로 설정하고 싶다면 xticks나 yticks 명령을 사용

lim : xlim, ylim 
- 플롯 그림을 보면 몇몇 점들은 그림의 범위 경계선에 있어서 잘 보이지 않는 경우가 있을 수 있다
- 그림의 범위를 수동으로 지정하려면 xlim 명령과 ylim 명령을 사용한다
- 이 명령들은 그림의 범위가 되는 x축, y축의 최솟값과 최댓값을 지정한다
- 현재 y 값 표현하는데에 최솟값을 -1로 지정했고 최댓값은 n_features 즉, 30개를 지정했다

'''