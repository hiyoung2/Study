# Chapter 7 Numpy

# 7.1 Numpy 개요
# 7.1.1 Numpy가 하는 일
# NumPy(넘파이)는 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 기본 라이브러리
# 라이브러리란 외부에서 읽어 들이는 파이썬 코드 묶음
# 라이브러리 안에는 여러 모듈이 포함, 모듈은 많은 함수가 통합된 것
# ex) 라이브러리 : numpy
#     모듈 : numpy.random
#     함수 : randit()

# 그 밖에 자주 이용되는 라이브러리는 SciPy, Pandas, scikit-learn, Matplotlib 등이 있다
# 파이썬이 머신러닝 분야에서 널리 활용되는 이유는 NumPy 등 과학 기술 계산에 편리한 라이브러리가 충실하기 때문
# 라이브러리와 개발 환경 등을 포함해 파이썬 환경 전체를 생태계(ecosystem)라고 부른다

# 7.1.2 NumPy의 고속 처리 경험
# 파이썬은 벡터와 행렬 계산 속도가 느리다
# 이를 보완하는 라이브러리가 NumPy이다

import numpy as np
import time
from numpy.random import rand

N = 150 # 행, 열의 크기

# 행렬 초기화
matA = np.array(rand(N, N))
matB = np.array(rand(N, N))
matC = np.array([[0] * N for _ in range(N)])

# 파이썬의 리스트를 사용하여 계산 
# 시작 시간 저장
start = time.time()

# for 문을 사용하여 행렬 곱셈 실행
for i in range(N) :
    for j in range(N) :
        for k in range (N) :
            matC[i][j] = matA[i][k] * matB[k][j]
print("파이썬 기능만으로 계산한 결과: %.2f[sec]" % float(time.time() - start))

# Numpy를 사용하여 계산
# 시작 시간 저장
start = time.time()

# NumPy를 사용하여 행렬 곱셈 시작
matC = np.dot(matA, matB)

print("NumPy를 사용하여 계산한 결과: %.2f[sec]" %float(time.time() - start))

'''
파이썬 기능만으로 계산한 결과: 2.99[sec]
NumPy를 사용하여 계산한 결과: 0.00[sec]
'''

