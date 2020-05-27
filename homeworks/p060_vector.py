# 4장 선형대수

# 선형대수는 벡터 공간을 다루는 수학의 한 분야
# 선형대수 모든 것을 알기엔 당장 이 책으로 어려움
# 핵심적인 개념 짚고 간다

# 4.1 벡터
# 간단히 말하면 vector, 벡터는 벡터끼리 더하거나 상수(scalar)와 곱해지면
# 새로운 벡터를 생성하는 개념적인 도구이다
# 더 자세하게는 벡터는 어떤 유한한 차원의 공간에 존재하는 점들이다!
# 대부분의 데이터, 특히 숫자로 표현된 데이터는 벡터로 표현할 수 있다
# 수많은 사람들의 키, 몸무게, 나이에 대한 데이터가 주어졌다고 해보자
# 그렇다면 주어진 데이터를(키, 몸무게, 나이)로 구성된 3차원의 벡터로 표현할 수 있을 것이다
# 또 다른 예로, 시험을 네 번 보는 수업을 가르친다면 각 학생의 성적을 
# (시험1 점수, 시험2 점수, .시험3 점수, 시험4 점수)로 구성된 4차원 벡터로 표현할 수 있을 것이다

# 벡터를 가장 간단하게 표현하는 방법은 여러 숫자의 리스트로 표현하는 것이다
# 예를 들어 3차원의 벡터는 세 개의 숫자로 구성된 리스트로 표현할 수 있다

# 앞으로 벡터는 float 객체를 갖고 있는 리스트인 Vector라는 타입으로 명시할 것이다

from typing import List
Vector = List[float]

height_weight_age = [  70,  # 인치
                      170,  # 파운드
                       40]  # 나이

grades = [95,    # 시험1 점수
          80,     # 시험2 점수
          75,    # 시험3 점수
          62]    # 시험4 점수

# 앞으로 벡터에 대한 산술 연산(arithmetic)을 하고 싶은 경우가 생길 것이다
# 파이썬 리스트는 벡터가 아니기 때문에 이러한 벡터 연산을 해주는 기본적인 도구가 없다
# 그러니 벡터 연산을 할 수 있게 해주는 도구를 직접 만들어 보자

# 종종 여기서는 두 개의 벡터를 더할 것이다
# 두 개의 벡터를 더한다는 것은 각 벡터상에 같은 위치에 있는 성분끼리 더하는 것이다
# 가령 길이가 같은 v와 w라는 두 벡터를 더한다면 계산된 새로운 벡터의 
# 첫 번째 성분은 v[0] + w[0], 두 번째 성분은 v[1] + w[1] 등등으로 구성된다
# 만약! 두 벡터의 길이가 다르다면 두 벡터를 더할 수 없다

# 예를 들어 [1, 2]로 구성된 벡터와 [2, 1]로 구성된 벡터를 더한다면
# [1+2, 2+1] 즉, [3, 3]으로 구성된 벡터가 계산된다

# 벡터 덧셈은 zip을 사용해서 두 벡터를 묶은 뒤 각 성분끼리 더하는 리스트 컴프리헨션을 사용하면 된다


def add(v: Vector, w: Vector):
    # 각 성분끼리 더한다
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add ([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

# 비슷하게 벡터 뺄셈은 각 성분끼리 빼 준다
def subtract(v: Vector, w: Vector):
    # 각 성분끼리 뺀다
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]
assert subtract ([5, 7, 9], [4, 5, 6] == [1, 2, 3])

# 또한 가끔씩 벡터로 구성된 리스트에서 모든 벡터의 각 성분을 더하고 싶은 경우도 있을 것이다
# 즉, 새로운 벡터의 첫 번째 성분은 모든 벡터의 첫 번째 성분을 더한 값, 
# 두 번째 성분은 모든 벡터의 두 번째 성분을 더한 값으로 구성된다

def vector_sum(vectors: List[Vector]):
    # 모든 벡터의 각 성분들끼리 더한다
    # vectors가 비어 있는지 확인
    assert vectors, "no vectors provided!"

    # 모든 벡터의 길이가 동일한 지 확인
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different size!"

    # i 번째 결괏값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

# 또한 벡터에 스칼라를 곱해줄 수 있어야 한다
# 스칼라 곱셈은 벡터의 각 원소마다 스칼라 값을 곱해 주는 방법으로 간단히 구현

def scalar_multiply(c: float, v: Vector):
    # 모든 성분을 c로 곱하기
    return [c * v_i for v_i in v]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# 이제 같은 길이의 벡터로 구성된 리스트가 주어졌을 때 각 성분별 평균을 구할 수도 있다

def vector_mean(vectors: List[Vector]):
    # 각 성분별 평균을 계산
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

# 벡터의 내적(dot product)은 조금 덜 자명하다
# 내적은 벡터의 각 성분별 곱한 값을 더해준 값이다

def dot(v: Vector, w: Vector):
    # v_i * w_i + ... + v_n * w_n
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1*4 + 2*5 + 3*6

# 만약 벡터 w의 크기가 1이라면 내적은 벡터 v가 벡터 w  방향으로 
# 얼마나 멀리 뻗어나가는지를 나타낸다
# 다른 관점에서 보자면 내적은 v가 w로 투영된 벡터의 길이를 나타낸다

# 내적의 개념을 사용하면 각 성분의 제곱 값의 합을 쉽게 구할 수 있다
def sum_of_squares(v:Vector):
    # v_1 * v_1 +... + v_n * v_n
    return dot(v, v)
assert sum_of_squares([1, 2, 3]) == 14 # 1*1 + 2*2 + 3*3


# 제곱 값의 합을 이용하면 벡터의 크기를 계산할 수 있다

import math

def magnitude(v:Vector):
    # 벡터 v의 크기를 반환
    return math.sqrt(sum_of_squares(v))    # math.sqrt는 제곱근을 계산해 주는 함수

assert magnitude([3, 4] == 5)

# 두 벡터 간의 거리를 계산하기 위해 필요한 것이 모두 준비되었다
# 두 벡터 간의 거리는 다음과 같이 정의한다
# 루트 (v1-w1)^2 + ... + (vn-wn)^2

# 코드로 구현
def squared_distance(v: Vector, w: Vector):
    # (v_1 - w_1)**2 + ... + (v_n - w_n)**2
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector):
    # 벡터 v와 w 간의 거리를 계산
    return math.sqrt(squared_distance(v, w))

# 다음과 같이 수정하면 더 깔끔
# def distance(v: Vector, w: Vector):
#     return magnitude(subtract(v, w))

