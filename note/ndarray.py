# <ndarray class>

# ndarray는 Numpy의 핵심인 다차원 행렬 자료구조 클래스이다
# 실제로 파이썬이 제공하는 List 자료형과 동일한 출력 형태를 갖는다
# ndarray는 다음과 같이 선언해 볼 수 있다

import numpy as np   
a = np.array([0, 1, 2, 3 ,4 , 5, 6, 7, 8, 9])  
b = np.array([0.12, 0.34, 0.45, 0.66])  
c = np.array(['a', 'b', 'c'])  
print(type(a))  # <class 'numpy.ndarray'>

# print(type(a))의 결과는 일반 파이썬 리스트와 동일하다
# * Python List
# - 여러 가지 타입의 원소
# - linked List 구현
# - 메모리 용량이 크고 속도가 느림
# - 벡터화 연산 불가

## Numpy ndarray 
# - 동일 타입의 원소
# - contiguous memory layout
# - 메모리 최적화, 계산 속도 향상
# - 벡터화 연산 가능

# 파이썬의 List로도 다차원 행렬 구현이 가능하고 
# List의 여러 함수를 사용해서 행렬 연산을 구현할 수 있다
# 그런데 왜 Numpy의 ndarray를 사용할까? 
# Numpy만의 선형대수 연산이 일반 파이썬의 List 보다 선형대수에 유리한 점이 많기 때문

## ndarray 활용하기
# ndarray의 기본 메소드 중 ndim, shape는 머신러닝의 텐서 구조와 차원을 알아내는 것과 동일

print(a.ndim)    # 1  
print(a.shape)   # (10, )

## 다차원 행렬 인덱싱 및 슬라이싱
# ndarray는 일반 파이썬 list에는 없는 콤마 사용, 인덱싱, 슬라이싱 가능

a = np.array([[0, 1, 2], [3, 4, 5]])  
a [0, 0]            # first row, first col  
print (a [0, 0])    # 0
a [0, 1]            # first row, second col  
print (a [0, 1])    # 1
a [-1, -1]          # last row, last col  
print(a [-1, -1])   # 5

# 위의 코드처럼 인덱싱이 가능   
# 인덱스로 생각하기!!!!!!!!

a = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
a[0,:]               # first row(전체)
print(a[0,:])        # [0 1 2 3]
a[:, 1]              # second col(전체)
print(a[:, 1])       # [1 5]
a[1, 1:]             # second row의 second col 부터끝까지
print(a[1, 1:])      # [5 6 7]
a[:, 0:3]            # 모든 row의 0:3까지 col
print(a[:, 0:3])     # [[0 1 2] [4 5 6]]