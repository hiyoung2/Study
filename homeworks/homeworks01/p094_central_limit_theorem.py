# 6.7 중심극한정리

# 정규분포가 중요한 이유 중의 하나는 중심극한정리(central limit theorem) 때문이다
# 중심극한정리란 동일한 분포에 대한 독립적인 확률변수의 평균을 나타내는 확률변수가 
# 대략적으로 정규분포를 따른다는 정리다

# 중심극한정리를 보다 더 쉽게 이해하기 위해 이항 확률변수(binomial random variable)를 예시로 보자
# 이항 확률변수는 n과 p 두 가지 파라미터로 구성되어 있다
# 이항 확률변수는 단순히 n개의 독립적인 베르누이 확률변수(Bernoulli random variable)를 더한 것이다
# 각 베르누이 확률변수의 값은 p의 확률로 1, 1 - p 의확률로 0이 된다
'''
import matplotlib.pyplot as plt
import math, random

def bernoulli_trial(p: float) : 
    # p의 확률로 1을, 1-p의 확률로 0을 반환
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) :
    # n개 bernoulli(p)의 합을 반환
    return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int):
    # binomial(n, p)의 결괏값을 히스토그램으로 표현
    data = [binomial(n, p) for _ in range(num_points)]

    # 이항분포의 표본을 막대 그래프로 표현
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color = '0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1-p))

    # 근사된 정규분포를 라인 차트로 표현
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf( i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distrbution vs. Normal Approximation")
    plt.show()

make_hist(0.75, 100, 10000)
'''