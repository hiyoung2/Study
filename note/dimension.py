# Scalar
# 숫자, 상수

# Vector
# Scalar들의 모임, 하나의 배열

# Matrix
# 2차원의 배열

# Tensor
# 3차원 이상의 배열

'''
1. 
[A B C D E F G H I ] (10, )
: scalar 10개, vector 1개 (1차원)

2.
[A B C D E]
[F G H I J] (2, 5) 
: 2행 5열, 행렬! Matrix (2차원)

3. 
[A B]
[C D]
[E F]
[G H]
[I J] (5, 2)
: 5행 2열, 행렬! Matrix (2차원)

4.
[A
B
C
D
E
F
G
H
I
J
] (10, 1)
: 10개의 행, 1개의 열로 이루어진 행렬(2차원)

=========================================
1.
[A B C D]
[E F G H] (2, 4)

2.
[A][B]
[C][D]
[E][F]
[G][H] (4, 2, 1)

1과 2는 무엇이 같을까? 변하지 않는 것은 뭐?
순서 AND 값이 안 바뀜
Scalar의 수도 같음

shape는 다르나, 데이터 값은 같다, 순서는 같다
2 * 4 = 4 * 2 * 1
reshape가 가능하다
'''

'''
# 차원 정리

차원         구조           input_shape       예          Layer
2차        1000 3           1차원            (3, )        Dense

3차       1000 3 1          2차원            (3, 1)       LSTM
      batch/time/feature 

4차      1000 28 28 1       3차원            (28, 28, 1)  Conv2D
'''