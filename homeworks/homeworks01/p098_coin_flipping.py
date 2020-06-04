# 7장 : 가설과 추론

# 7. 1 통계적 가설 검정
# 특정 가설이 사실인지 아닌지 검정해 보고 싶은 경우가 있다
# 여기서 가설(hypothesis)이란 
# '이 동전은 앞뒤가 나올 확률이 공평한 동전이다' 
# '데이터 과학자는 R보다 파이썬을 선호한다'
# '닫기 버튼이 작아서 찾기 힘든 광고 창을 띄우면 사용자는 해당 사이트를 다신 들어가지 않을 것이다'
# 등과 같은 주장을 의미, 데이터 통계치에 대한 이야기로 변환될 수 있다

# 그런 통계치들은 다양한 가정하에서 특정 분포에 대한 확률변수의 관측치로 이해할 수 있고
# 그런 가정들이 얼마나 타당한지 알 수 있게 해주기도 한다

# 고전적인 가설검정에는 기본 입장을 나타내느 귀무가설(H0, null hypothesis)과
# 대비되는 입장을 나타내는 대립가설(H1, alternative hypothesis)을 통해 통계적으로 비교해
# 귀무가설을 기각할지 말지를 결정한다

# 7.2 예시 : 동전 던지기
# 동전이 하나 있다 우리는 이 동전이 공편한지 아닌지 검정하고 싶다
# 이 동전에서 앞면이 나올 확률이 p라고 한다면 동전이 공평하다는 의미인 
# 'p == 0.5 이다'는 귀무가설,
# 'p != 0.5이다' 는 대립가설이 된다
# 동전을 n번 던져서 앞면이 나온 횟수를 X를 세는 것으로 검정을 진행해보자
# 동전 던지기는 각각 베르누이 분포를 따를 것이며 이는 X가 이항분포를 따르는 확률변수라는 것을 의미
# 이항분포는 정규화 분포로 근사할 수 있다

from typing import Tuple
import math
def normal_approximation_to_binomial(n: int, p: float) :
    # Binomial(n, p)에 해당되는 mu(평균)와 sigma(표준편차) 계산
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# 확률변수가 정규분포를 따른다는 가정하에 normal_cdf를 사용하면
# 실제 동전 던기로부터 얻은 값이 구간 안(혹은 밖)에 존재할 확률을 계산할 수 있다


'''
# scratch????
from scratch.probability import normal_cdf

# 누적 분포 함수는 확률변수가 특정 값보다 작을 확률을 나타낸다
normal_probability_below = normal_cdf

# 만약 확률변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 것을 의미
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1):
    # mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률
    return 1 - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 hi보다 작고 lo보다 작지 않다면 확률변수는 hi와 lo 사이에 존재
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1):
    # mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않다는 것을 의미
def normal_probability_outisde(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1):
    # mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 없을 확률
    return 1 - normal_probability_between(lo, hi, mu, sigma)

# 반대로 확률이 주어졌을 때 평균을 중심으로 하는 (대칭적인) 구간을 구할 수도 있다
# 예를 들어 분포의 60%를 차지하는 평균 중심의 구간을 구하고 싶다면 
# 양쪽 꼬리 부분이 각각 분포의 20%를 차지하는 지점을 구하면된다

from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1):
    # P(Z <= z) - probability인 z값을 반환
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1):
    # P(Z >= z) = robability인 z값을 반환
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1):
    # 입력한 probability 값을 포함하고
    # 평균을 중심으로 대칭적인 구간을 반환
    tail_probability = (1 - probability) / 2

    # 구간의 상한 tail_probability 값 이상의 확률 값을 갖고 있다
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # 구간의 하한은 tail_probability 값 이하의 확률 값ㅇ르 갖고 있다
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

# 실제로 동전을 1000번 던지자 (n=1000)
# 만약 동전이 공평하다는 가설이 맞다면 
# X는 대략 평균이 500이고 표준편차가 15.8인 정규분포를 따를 것이다

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# 이제 제1종 오류를 얼마나 허용해 줄 것인지를 의미하는 유의수준(significance)를 결정해야함
# 제1종 오류란 비록 H0이 참이지만 H0을 기각하는 'false positive(가양성)' 오류를 의미
# 유의수준은 보통 5%나ㅏ 1%로 설정하는데 여기서는 5%로 하자
# 다음의 코드에서 X가 주어진 범위를 벗어나면 귀무가설 H0을 기각하는 가설 검정을 고려해보자

# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p가 정말로 0.5, 즉 H0이 참이라면 X가 주어진 범위를 벗어날 확률은 우리가 원하는 대로 5% 밖에 안 됨
# 바꿔 말하면 H0가 참이라면 이 가설검정은 20번 중 19번은 올바른 결과를 줄 것

# 한편 제2종 오류를 범하지 않을 확률을 구하려면 검정력(power)을 알 수 있다
# 제2종 오류란 H0가 거짓이지만 H0를 기각하지 않는 오류를 의미하기 때문에,
# 제2종 오류를 측정하기 위해서는 H0가 거짓이라는 것이 무엇을 의미하는지를 알아야 한다
# p가 0.5가 아니라는 말은 X의 분포에 관해 많은 것을 알려주지는 않는다
# 예를 들어 p가 0.55, 즉 동전의 앞면이 나올 확률이 약간 편향되어 있다면 검정력은 다음과 같다

# p가 0.5라고 가정할 때, 유의수준이 5%인 구간
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p = 0.55인 경우의 실제 평균과 표준 편차
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 제2종 오류란 귀무가설(H0)을 기각하지 못한다는 의미
# 즉, X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1- type_2_probability # 0.887

# 한편 p<=0.5 즉 동전이 앞면에 편향되지 않을 경우를 귀무가설로 정한다면 X가 500보다 크면
# 귀무가설을 기각하고, 500보다 작다면 기각하지 않는 단측검정(one-sided-test)이 필요해짐
# 유의수준이 5%인 가설검정을 위해서는 normal_probability_below를 사용하여
# 분포의 95%가 해당 값 이하인 경계 값을 찾을 수 있다

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# 결괏값은 526( < 531, 분포 상위 부분에 더 높은 확률을 주기 위해)

type_2_probability - normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936

# 이 가설검정은 더 이상 X가 469보다 작을 때 H0을 기각하는 게 아니라
# (H1이 참이라면 이는 거의 발생하지 않을 것)
# X가 526에서 531 사이일 때 H0을 기각하기 때문에 
# (H1이 참이라면 이는 가끔 발생할 것)
# 전보다 검정력이 더 좋아졌다고 볼 수 있다
'''