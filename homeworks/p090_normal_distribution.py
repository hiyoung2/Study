# 6.6 정규분포
# 정규분포는 유명한 종형 곡선 모양의 분포
# 평균인 뮤와 표준편차 시그마의 두 파라미터로 정의된다
# 평균은 종의 중심이 어디인지를 나타내며 표준편차는 종의 폭이 얼마나 넓은지를 나타낸다
# 정규분포의 밀도 함수는 다음과 같이 구현할 수 있다

import math

SQRT_TWO_PI  = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1):
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma = 1) for x in xs], '-' , label= 'mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma = 2) for x in xs], '--', label = 'mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma = 0.5) for x in xs], ':', label = 'mu=1, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label = 'mu= -1, sigma = 1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

# 정규분의 누적 분포 함수를 간단하게 표현하기는 어렵지만 파이썬의 math.erf를 사용하면 가능해진다
# math.erf : 오차 함수

def normal_cdf(x: float, mu: float = 0, sigma: float = 1):
    return (1 + math.erf((x - mu)/ math.sqrt(2) / sigma)) / 2

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_cdf(x, sigma = 1) for x in xs], '-' , label= 'mu=0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma = 2) for x in xs], '--', label = 'mu=0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma = 0.5) for x in xs], ':', label = 'mu=1, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label = 'mu= -1, sigma = 1')
plt.legend()
plt.title("Various Normal cdfs")
plt.show()

# 가끔씩 특정 확률을 갖는 확류변수의 값을 찾기 위해 normal_cdf의 역함수가 필요할 수도 있다
# 누적 분포 함수의 역함수를 쉽게 계산하는 방법은 없지만
# 누적 분포 함수가 연속 및 증가 함수라는 점을 고려하면 이진 검색을 사용해 비교적 쉽게 값을 구할 수 있다

'''
def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001):
                       # 이진 검색을 사용해서 역함수를 근사
                       # 표준 정규화 분포가 아니라면 표준 정규 분포로 변환
                       if mu != 0 or sigma != 1:
                           return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
                        low_z = -10.0                      # normal_cdf(-10)은 0에 근접
                        
                        hi_z  = 10.0                       # normal_cdf(10)은 1에 근접
                        while hi_z - low_z > tolerance:
                            mid_z = (low_z + hi_z) / 2      # 중간 값
                            mid_p = normal_cdf(mid_z)       # 중간 값의 누적분포 값을 계산
                            if mid_p < p:
                                low_z = mid_z              # 중간값이 너무 작다면 더 큰 값들을 검색
                            else:
                                hi_z = mid_z               # 중간 값이 너무 크다면 더 작은 값들을 검색
                        return mid_z
'''

# 앞의 함수는 원하는 확률에 가까워질 때까지 표준정규분포의 구간을 반복적으로 이등분한다                    