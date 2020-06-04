# 5.1.2 산포도
'''
# 산포도, dispersion 는 데이터가 얼마나 퍼져 있는지를 나타낸다
# 보통 0과 근접한 값이면 데이터가 거의 퍼져 있지 않다는 의미, 
# 큰 값이면 매우 퍼져 있다는 것을 의미하는 통계치다
# 예를 들어 가장 큰 값과 가장 작은 값의 차이를 나타내는 범위는 산포도를 나타내는 가장 간단한 통계치이다

# 파이썬에서 range는 이미 다른 것을 의미, 다른 이름을 사용!

# num_friends = 

from typing import List
def data_range(xs: List[float]):
    return max(xs) - min(xs)

assert data_range(num_friends) == 99

# 범위는 max, min이 같은 경우에 0이 된다
# 이 경우 x의 데이터 포인트는 모두 동일한 값을 갖고 있으며 데이터가 퍼져 있지 않다는 것을 의미
# 반대로 범위의 값이 크다면 max가 min에 비해 훨씬 크다는 것을 의미
# 데이터가 더 퍼져 있다는 것을 의미

# 범위 또한 중앙값처럼 데이터 전체에 의존하지 않는다
# 모두 0 혹은 100으로 구성된 데이터나 0, 100 그리고 수많은 50으로 구성된 데이터나 동일한 범위를 갖게 된다

# 하지만 첫 번째 데이터(0 혹은 100으로만 구성)가 더 퍼져 있다는 느낌이 든다

# 분산(variance)은 산포도를 측정하는 약간 더 복잡한 개념이며 다음과 같이 계산

from scratch.linear_algebra import sum_of_squares

def de_mean(xs: List[float]):
    # x의 모든 데이터 포인트에서 평균을 뺌(평균을 0으로 만들기 위해)
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]):
    # 편차의 제곱의 (거의)평균
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

assert 81.54 < variance(num_friends) < 81.55

# 식을 살펴보면 편차의 제곱의 평균을 계산하는데, n 대신에 n-1로 나누는 것을 확인할 수 있다
# 이는 편차의 제곱 합을 n으로 나누면 bias 때문에 
# 모분산에 대한 추정값이 실제 모분산보다 작게 계산되는 것을 보정하기 위한 것

# 데이터 포인트의 단위( 예시 같은 경우 , '친구 수')가 무엇이든 간에
# 중심 경향성은 같은 단위를 가진다
# 범위 또한 동일한 단위
# 하지만 분산의단위는 기존 단위의 제곱이다 (즉, 친구 수의 제곱)
# 그렇기 때문에 분산 대신 원래 단위와 같은 단위를 가지는
# 표준 편차(standard deviation)을 이용할 때가 더 많다

import math

def standard_deviation(xs: List[float]):
    # 표준편차는 분산의 제곱근
    return math.sqrt(variance(xs))

assert 9.02 < standard_deviation(num_friends) < 9.04
'''