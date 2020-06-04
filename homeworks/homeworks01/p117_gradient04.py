# 8.5 경사 하강법으로 모델 학습

# 경사 하강법으로 데이터에 적합한 모델의 파라미터를 구할 것!
# 대부분의 경우 데이터셋과 (미분 가능한) 하나 이상의 파라미터로 구성된 모델이 주어질 것
# 손실함수를 통해 모델이 얼마나 주어진 데이터에 적합한지 계산할 것

# 주어진 데이터가 더 이상 변하지 않는다고 가정하면 손실 함수는 모델의 파라미터가
# 얼마나 좋고 나쁜지 알려준다
# 경사 하강법으로 손실을 최소화하는 모델의 파라미터를 구할 수 있다는 것!

# x는 -50 ~ 49 사이의 값이며 y는 항상 20 * x + 5
from typing import List
import math

Vector = List[float]

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

# 이 경우에는 x와 y가 선형관계라는 것을 이미 알고 있지만 데이터를 통해서 이 관계를 학습시켜보자,,,
# 경사 하강법으로 평균제곱오차(mse)를 최소화해주는 경사와 절편을 구해보자
# 먼저 한 개의 데이터 포인트에서 오차의 그래디언트를 계산해주는 함수를 만들자
'''
def linear_gradient(x: float, y: float, theta: Vector):
    slope, intercept = theta
    predicted = slope * x + intercept         # 모델의 예측값
    error = (predicted - y)                   # 오차는 (예측값 - 실젯값)
    squared_error = error ** 2                # 오차의 제곱을 최소화하자
    grad = [2 * error * x, 2 * error]       # 그래디언트를 사용한다
    return grad
'''
# 다음의 경사 하강법 적용
# 1) 임의의 theta로 시작
# 2) 모든 그래디언트의 평균을 게산
# 3) theta를 2번에서 계산된 값으로 벼경
# 4) 반복

'''
from scratch.linear_algebra import vector_mean

# 임의의 경사와 절편으로 시작
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    # 모든 그래디언트의 평균을 게산
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # 그래디언트만큼 이동
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1,      "slope should be about 20"
assert 4.9 < intercept < 5.1 ,   "intercept should be about 5"
'''
