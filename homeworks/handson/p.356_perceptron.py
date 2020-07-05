# 10.1.3 퍼셉트론
# 퍼셉트론, perceptron은 가장 간단한 인공 신경망 구조 중 하나
# 1957년, 프랑크 로젠블라트가 제안
# 퍼셉트론은 TLU(threshold logic unit) 또는 LTU(linear threshold unit)으로 불리는
# 조금 다른 형태의 인공 뉴런을 기반으로 한다
# TLU(LTU)는 입력과 출력이 (이진 on/off) 어떤 숫자이고, 각각의 입력 연결은 가중치와 연결되어 있다
# TLU는 입력의 가중치 합을 계산한 뒤 계산된 합에 계단 함수를 적용하여 결과를 출력한다
# 퍼셉트론에서 가장 널리 사용되는 계단 함수는 헤비사이드 계단 함수(heaviside step function)이다
# 이따금 부호 함수 sign function을 대신 사용하기도 함

# 퍼셉트론에서 일반적으로 사용하는 계단 함수(임곗값을 0으로 가정)
'''
heaviside(z) = 0 (z < 0)   ,  1 (z >= 0)

sgn(z) = -1 (z < 0)   ,  0 (z == 0)  ,  +1 (z  > 0)
'''

# threshold : 문턱값, 임계값, 역치... 등으로 불림
# 어떤 반응을 일으키기 위해 요구되는 최소한의 자극의 세기
# 쉽게 말하면 자극의 세기가 threshold 값을 넘기면 반응을 일으키고
# threshold 값을 넘기지 못하면 반응이 일어나지 않음


# 하나의 TLU는 간단한 선형 이진 분류 문제에 사용할 수 있다
# 입력의 선형 조합을 계산해서 그 결과가 임곗값을 넘으면 양성 클래스를 출력, 그렇지 않으면 음성 클래스를 출력

# 퍼셉트론은 층이 하나뿐인 TLU로 구성된다 
# 각 TLU는 모든 입력에 연결되어 있다
# 한 층에 있는 모든 뉴런이 이전 층의 모든 뉴런과 연결되어 있을 때 이를 완전 연결층(fully connected layer) 또는 밀집층(dense layer)라고 부른다
# 퍼셉트론의 입력은 입력 뉴런 input neuron이라 불리는 특별한 통과 뉴런에 주입된다
# 이 뉴런은 어떤 입력이 주입되든 그냥 출력으로 통과시킨다
# 입력층, input layer는 모두 입력 뉴런으로 구성된다
# 보통 거기에 편향 특성이 더해진다
# 전형적으로 이 편향 특성은 항상 1을 출력하는 특별한 종류의 뉴런인 편향 뉴런, bias neuron으로 표현 된다


# 사이킷런은 하나의 TLU 네트워크를 구현한 Peceptron 클래스를 제공
# 사이킷런은 sklearn.neural_network 아래에 회귀와 분류의 다층 퍼셉트론 구현인 
# MLPClassifier와 MLPRegressor도 제공한다

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris  = load_iris()
x = iris.data[:, (2, 3)] # 꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int) # 부채붓꽃(Iris Setosa)인가?

per_clf = Perceptron()
per_clf.fit(x, y)

y_pred = per_clf.predict([[2, 0.5]])

# Perceptron Class는 매개변수가 
# loss = "perceptron", learning_rate = "constant", eta0 = 1(학습률), penalty = None(규제없음)인 SGDClassifier와 같다
# 로지스틱 회귀 분류기와 달리 퍼셉트론은 클래스 확률을 제공하지 않으며 고정된 임곗값을 기준으로 예측을 만든다
# 따라서 퍼셉트론보다 로지스틱 회귀가 선호됨

# 그러나
# 퍼셉트론은 일부 간단한 문제를 풀 수 없다
# 대표적으로 XOR 배타적 논리합 분류 문제
# 이것에 대해서는 퍼셉트론을 여러 개 쌓아 올리면 일부 제약을 줄일 수 있다는 사실이 밝혀졌는데
# 이런 인공 신경망을 다층 퍼셉트론 , MLP라고 한다
# MLP는 XOR문제를 풀 수 있다
