# 4.2 행렬
# 행렬, matrix는 2차원으로 구성된 숫자의 집합
# 리스트의 리스트로 표현 가능 [[]]
# 리스트 안의 리스트들은 행렬의 행(row)을 나타내며 모두 같은 길이를 가지게 된다
# 예를 들어 A라는 행렬에서 A[i][j]는 i번째 행, j번째 열에 속한 숫자를 의미
# 수학의 관습에 따라 행렬은 대문자로 표기

# 타입 명시를 위한 별칭
from typing import List

Matrix = List[List[float]]

A = [[1, 2, 3],       # A는 2개의 행과 3개의 열로 구성
    [4, 5, 6]]

B = [[1, 2],
     [3, 4,],
     [5, 6,]]         # B는 3ㅐ의 행과 2개의 열로 구성

# 수학에서는 첫 번째 행을 행1, 첫 번째 열을 열1 이라고 표기
# but 파이썬의 리스트는 0부터 시작하기 때문에 여기서도 첫 번째 행을 행0, 첫 번째 열을 열0으로 표기함

# 행렬을 리스트의 리스트로 나타내는 경우 행렬 A는 len(A)개의 행과 len(A[0])개의 열로 구성되어 있는 것

from typing import Tuple
def shape(A: Matrix):
    # (열의 갯수, 행의 갯수)를 반환
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # 첫 번째 행의 원소의 갯수 
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3) # 2행 3열

# 행렬이 n개의 행과 k개의 열로 구성되어 있다면 이 행렬을 'n x k' 행렬이라 부르자
# n x k 행렬에서 각 행의 길이는 k이고 각 열의 길이는 n이다

def get_row(A: Matrix, i: int):
    # A의 i번째 행을 반환
    return A[i]                # A[i]는 i번째 행을 나타냄

def get_column(A: Matrix, j : int):
    # A의 j번째 열을 반환
    return [A_i[j]             # A_i 행의 j번째 원소
            for A_i in A]      # 각 A_i 행에 대해

# 이제 shape, 형태가 주어졌을 때 각 형태에 맞는 행렬을 생성하고
# 각 원소를 채워 넣는 함수를 만들어 보자
# 중첩된 리스트 컴프리헨션을 사용해서 만든다

from typing import Callable
def make_matrix(num_rows: int, 
                num_cols: int,
                entry_fn: Callable[[int, int], float]):
    # (i, j) 번째 원소가 entry_fn(i, j)인
    # num_rows x num_cols 리스트를 반환
    return [[entry_fn(i,j)             # i가 주어졌을 때, 리스트를 생성
            for j in range(num_cols)]  # [entry_fn(i, 0), ...]
            for i in range(num_rows)]  # 각 i에 대해 하나의 리스트를 생성

# 이 함수를 사용해서 다음과 같은 5 x 5 단위 행렬(identity matrix, 대각선의 원소는 1, 나머지 원소는 0인 경우)
# 를 생성할 수 있다

def identity_matrix(n: int):
    # 단위 행렬을 반환
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1,0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

# 앞으로 행렬은 몇 가지 이유로 매우 중요해짐
# 먼저 각 벡터를 행렬의 행으로 나타냄으로써 여러 벡터로 구성된 data set을 행렬로 표현할 수 있음
# 예를 들어 1000명의 키, 몸무게, 나이가 주어졌다면 1000 x 3 행렬로 표현이 가능

# data  = [[70, 170. 40], 
#          [65, 120, 26],
#          [77, 250, 19]]

# 두 번째로 나중에 더 자세히 다루겠지만 k차원의 벡터를 n차원의 벡터로 변형해주는 선형 함수를
# n x k 행렬로 표현할 수 있다
# 세 번째로 행렬로 이진 관계(binary relationship)을 나타낼 수 있다
# 1장 '들어가기'에서 네트워크의 엣지(edge)들을 (i, j) 쌍의 집합으로 표현했다
# 이러한 네트워크의 구조를 행렬로 나타낼 수도 있다
# 예를 들어 i와 j가 연결되어 있다면 A[i], [j]의 값이 1이고 그렇지 않다면 0인 행렬 A로 네트워크 표현 가능

# 만약 네트워크 안에 연결된 사용자의 숫자가 적다면 행렬은 수많은 0값을 저장해야 하기 때문에
# 네트워크를 표현하기에 훨씬 더 비효율적
# BUT, 행렬에서는 두 사용자가 연결되어 있는지 훨씬 빠르게 확인이 가능
# 모든 엣지를 살펴보지 않고 직접 행렬의 값을 확인해보면 된다

# assert friend_matrix[0][2] == 1, "참, 사용자 0과 2는 친구이다"
# assert friend_matrix[0][8] == 0, "거짓, 사용자 0과 8은 친구가 아니다"

# 또 사용자가 누구와 연결되어 있는지 알아보기 위해서는 해당 사용자를 나타내는
# 열 또는 행만 보면 된다

# 하나의 행만 살펴보면 된다
# friends_of_five = [i
#                    for i, is_friend in enumerate(friend_matrix[5])
#                    if is_friend]

# 이러한 작업을 빠르게 처리하기 위해 각 사용자 객체에 해당 사용자와 연결된 사용자들을
# 리스트를 사용해 표현했다
# 하지만 네트워크의 크기가 커지거나 형태가 지속적으로 변한다면
# 이러한 방법은 매우 비효율적이고 관리하기가 힘들어질 것이다

