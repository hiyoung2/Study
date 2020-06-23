# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하시오
# 파일명 : m36_outliers2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i] 
        quartile_1, quartile_3 = np.percentile(data, [25, 75]) 
        print("1사분위 : ",quartile_1)                                       
        print("3사분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data_out > upper_bound) | (data_out < lower_bound))
        outliers.append(out)
    return outliers


a = np.array([[1, 10000], [500, 5], [2, 6], [3, 7], [4, 40], [9, 90]])
print(a.shape) # (6, 2)

b = outliers(a)
print(b)

'''
1사분위 :  2.25
3사분위 :  7.75
1사분위 :  6.25
3사분위 :  77.5
[(array([0, 1, 4, 5], dtype=int64), array([1, 0, 1, 1], dtype=int64)), (array([0, 1], dtype=int64), array([1, 0], dtype=int64))]
'''

c = np.array([[1, 2, 3, 100, 4, 5], [2, 4, 6, 20000, 8, 500]])
print(c.shape) # (2, 6)

d = outliers(c)
print(d)
'''
1사분위 :  1.25
3사분위 :  1.75
1사분위 :  2.5
3사분위 :  3.5
1사분위 :  3.75
3사분위 :  5.25
1사분위 :  5075.0
3사분위 :  15025.0
1사분위 :  5.0
3사분위 :  7.0
1사분위 :  128.75
3사분위 :  376.25
[(array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int64), array([2, 3, 4, 5, 1, 2, 3, 4, 5],...
'''

'''
# for pandas
def outliers(data_out):
    outliers = []
    for i in range(len(data_out.columns)):
        data = data_out.iloc[:, i]
        quartile_1 = data.quantile(.25)
        quartile_3 = data.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers
'''


# 리뷰
'''
def outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]): # data_out.shape[1]은 데이터의 컬럼
        data = data_out[:, i] # 하나 하나 컬럼이 분리된, data = for문 1번 : 1열, for문 2번 : 2열, ...
        quartile_1, quartile_3 = np.percentile(data, [25, 75]) 
        # percentile은 위치를 1/100 단위로 나눈 것

        # Numpy의 기술 통계 함수

        # 1. 데이터의 갯수 : len(x)
        # 2. 평균(통계용어로는 샘플 평균이라고 한다) : np.mean(x)
        # 3. 샘플 분산(데이터와 샘플 평균가의 거리의 제곱의 평균, 샘플 분산이 작으면 데이터가 모여 있는 것, 크면 흩어져 있는 것) : np.var(x)
        # 4. 샘플 표준편차(샘플 분산의 양의 제곱근 값) : np.std(x)
        # 5. 최댓값과 최솟값(최댓값은 데이터 중에서 가장 큰 값, 최솟값은 가장 작은 값) : np.max(x), np.min(x)
        # 6. 중앙값(데이터를 크기대로 정렬했을 때 가장 가운데에 있는 수, 만약 데이터의 수가 짝수이면 가장 가운데에 있는 두 수의 평균을 사용) : np.median(x)
        # 7. 사분위수(quartile, 데이터를 가장 작은 수부터 가장 큰 수까지 크기가 커지는 순서대로 정렬하였을 때 1/4, 2/4, 3/4 위치에 있는 수를 말함)
        #    (각각 1사분위수, 2사분위수, 3사분위수라고 한다, 1/4의 위치란 전체 데이터의 수가 만약 100개이면 25번째 순서, 즉 하위 25%를 말한다)
        #    (따라서 2사분위수는 중앙값과 같다, 때로는 위치를 1/100 단위로 나눈 백분위수, percentile을 사용하기도 한다)
        #    (1사분위수는 25% 백분위수와 같다)
        #    np.percentile(x, 0) : 최솟값
        #    np.percentiel(x, 25) : 1사분위 수
        #    np.percentile(x, 50) : 2사분위 수
        #    np.percentile(x, 75) : 3사분위 수
        #    np.percentile(x, 100) : 최댓값

        # 8. SciPy 패키지에는 여러가지 기술 통계 값을 한 번에 구해주는 describe 명령이 있다
        # from scipy.stats import describe
        # describe(x)
        # DescribeResult(nobs = ~, minmax = (~, ~), mean = ~, variance = ~, ....)

        print("1사분위 : ",quartile_1)                                       
        print("3사분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5) # 25% 지점에서 1.5를 
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers
'''