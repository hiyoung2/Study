# 7.2 NumPy 1차원 배열
# 7.2.1 import
# 본격적으로 NumPy를 사용한 프로그래밍 도전

# NumPy를 import 할 때는 import numpy로 표기
# import한 NumPy는 'numpy.모듈명' 형태로 사용
# 이 때 import numpy as np처럼 as를 사용하여 표기하면 패키지명을 변경 할 수 있다
# 'numpy.모듈명' 대신 'np.모듈명' 형태로 간단하게 사용할 수 있다
import numpy as np

# 7.2.2 1차원 배열
# NumPy에는 배열을 고속으로 처리하는 ndarray 클래스가 준비되어 있다
# ndarray를 생성하는 방법 중 하나는 NumPy의 np.array() 함수를 이용하는 것이다
# 'np.array(리스트)' 로 리스트를 전달하여 생성할 수 있다

np.array([1, 2, 3])

# np.arange() 함수를 이용하는 방법도 있다
# np.arange(X)로 표기하여 일정한 간격으로 증감시킨 값의 요소를 X개 만들어준다

np.arange(4) 
print(np.arange(4)) # [0 1 2 3]

# ndarray 클래스는 1차원의 경우 벡터,
# 2차원의 경우 행렬
# 3차원 이상은 텐서라고 한다
# 텐서는 수학적인 개념이지만, 머신러닝에서는 단순히 행렬의 개념을 확장한 것으로 봐도 좋음

# 1차원 ndarray 클래스
array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# 2차원 ndarray 클래스
array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# 3차원 ndarray 클래스
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


# 문제
import numpy as np
storages = [24, 3, 4, 23, 10, 12]
print(storages)

np_storages = np.array(storages)
print(type(np_storages)) # <class 'numpy.ndarray'>

# 7.2.3 1차원 배열의 계산
# 리스트에서는 요소별로 계산하기 위해 루프시킨 뒤 하나씩 더했지만
# ndarray에서는 루프를 사용하지 않아도 된다
# ndarray의 산술 연산은 같은 위치에 있는 요소끼리 게산된다

# NumPy 사용 X
storages = [1, 2, 3, 4]
new_storages = []
for n in storages :
    n += n
    new_storages.append(n)
print(new_storages) # [2, 4, 6, 8]

# NumPy 사용
import numpy as np
storages = np.array([1, 2, 3, 4])
storages += storages
print(storages) # [2 4 6 8]

# 문제
arr = np.array([2, 5, 3, 4, 8])


print('arr + arr')
print( arr + arr)

print('arr - arr')
print(arr - arr)

print('arr**3')
print(arr**3)

print('1 / arr')
print(1 / arr)


# 7.2.4 인덱스 참조와 슬라이스
# 리스트형과 마찬가지로 NumPy도 인덱스 참조와 슬라이스가 가능하다
# 1차원 배열은 벡터이므로 인덱스를 참조한 곳은 스칼라값(일반 정수와 소수점 등)이 된다

arr = np.arange(10)
print(arr) # [0 1 2 3 4 5 6 7 8 9]

arr[0:3] = 1 # 0, 1, 2 번째가 1이 된다
print(arr) # [1 1 1 3 4 5 6 7 8 9]

arr = np.arange(10)
print(arr)

print(arr[3:6]) # [3 4 5]

arr[3:6] = 24
print(arr) # [ 0  1  2 24 24 24  6  7  8  9]

# 7.2.5 ndarray 사용시 주의사항
# ndarray 배열을 다른 변수에 그대로 대입한 경우, 해당 변수의 값을 변경하면 
# 원래 ndarray 배열의 값도 변경된다(파이썬의 리스트와 동일)
# ndarray를 복사하여 두 개의 변수를 별도로 만들고 싶을 때는 copy() 메서드를 사용한다
# '복사할 배열.copy()'로 복사할 수 있다

arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1
arr2[0] = 100

print(arr1) #
# arr2 변수를 변경하면 원래 변수 arr1도 영향을 받음
# [1 2 3 4 5]
# [100   2   3   4   5]

# copy()를 이용해서 영향을 안 가게 해 보자

arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1.copy()
arr2[0] = 100

print(arr1) # [1 2 3 4 5] 
# arr2 변수를 변경해도 arr1에 영향 X

# 7.2.6 view와 copy
# 파이썬의 리스트와 ndarray의 차이는 ndarray의 슬라이스는 배열의 복사본이 아닌 view라는 점이다
# view란 원래 배열의 데이터를 가리키는 것이다(원본 참조)
# 즉, ndarray의 슬라이스는 원래 ndarray를 변경하게 된다
# 슬라이스를 복사본으로 만들려면 copy() 메서드를 사용하여 arr[:].copy()로 한다

arr_List = [x for x in range(10)]
print("리스트형 데이터입니다.")
print("arr_List: ", arr_List) # arr_List:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print()

arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

print("리스트의 슬라이스는 복사본이 생성되므로 arr_List에는 arr_List_copy의 변경이 반영되지 않는다")
print("arr_List: ", arr_List) # arr_List:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print()


# Numpy의 ndarray에 슬라이스를 이용한 경우
arr_NumPy = np.arange(10)
print("Numpy의 ndarray 데이터")
print("arr_Numpy: ", arr_NumPy) # [0 1 2 3 4 5 6 7 8 9]
print()

arr_NumPy_view = arr_NumPy[:]
arr_NumPy_view[0] = 100

print("NumPy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로 arr_NumPy_view를 변경하면 arr_Numpy에 반영된다")
print("arr_NumPy:", arr_NumPy) # [100   1   2   3   4   5   6   7   8   9]
print()

# NumPy의 ndarray에서 copy()를 사용한 경우
arr_NumPy = np.arange(10)
print("NumPy의 ndarray에서 copy()를 사용한 경우")
print("arr_Numpy: ", arr_NumPy)  # [0 1 2 3 4 5 6 7 8 9]
print()

arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100

print("copy()를 사용하면 복사본이 생성, arr_NumPy_copy는 arr_NumPy에 영향을 미치지 않는다")
print("arr_NumPy: ", arr_NumPy) # [0 1 2 3 4 5 6 7 8 9] / 100으로 변경한 게 영향 미치지 않음을 확인


# 7.2.7 부울 인덱스 참조
# 부울 인덱스 참조란 []안에 논리값(True/False) 배열을 사용하여 요소를 추출하는 방법을 말한다
# 'arr[ndarray 논리값 배열]'로 표기하면 논리값(부울) 배열의 True에 해당하는 요소의 ndarray를 만들어 반환해준다

arr = np.array([2, 4, 6, 7])
print(arr[np.array([True, True, True, False])]) # [2 4 6]

arr = np.array([2, 4, 6, 7])
print(arr[arr % 3 == 1]) # [4 7]

arr = np.array([2, 3, 4, 5, 6, 7])

# 부울 배열의 출력은 'print(조건)'으로 할 수 있다
print(arr % 2 == 0) # [ True False  True False  True False]

print(arr[arr % 2 ==0]) # [2 4 6]



# 7.2.8 범용 함수
# 범용 함수 , universal function는 ndarry 배열의 각 요소에 대한 연산 결과를 반환하는 함수
# 요소별로 계산하므로 다차원 배열ㄹ에도 사용 가능
# 범용 함수는 인수가 하나인 경우와 두 개인 경우가 있다

# 인수가 하나인 경우의 대표적인 예는 요소의 절댓값을 반환하는 np.abs(), 
# 요소의 e(자연 로그의 밑)의 거듭제곱을 반환하는 np.exp(),
# 요소의 제곱근을 반환하는 np.sqrt() 등이 있다

# 인수가 두 개인 경우의 대표적인 예는 요소 간의 합을 반환하는 np.add(),
# 요소 간의 차이를 반환하는 np.substract(),
# 요소 간의 최댓값을 저장한 배열을 반환하는 np.maximun() 등이 있다

arr = np.array([4, -9, 16, -4, 20])
print(arr)

arr_abs = np.abs(arr)
print(arr_abs)

print(np.exp(arr_abs))
# [5.45981500e+01 8.10308393e+03 8.88611052e+06 5.45981500e+01
#  4.85165195e+08]

print(np.sqrt(arr_abs))
# [2.         3.         4.         2.         4.47213595]

# 7.2.9 집합 함수
# 집합 함수란 수학의 집합 연산을 수행하는 함수
# 1차원 배열만을 대상으로 한다
# 대표적인 집합 함수로는 배열 요소에서 중복ㅇ르 제거하고 정렬한 결과를 반환하는 np.unique(),
# 배열 x와 y의 합집합을 정렬해서 반환하는 np.union1d(x, y),
# 배열 x와 y의 교집합을 정렬해서 반환하는 np.intersect1d(x, y) 함수,
# 배열 x에서 y를 뺀 차집합을 정렬해서 반환하는 np.setdiff1d(x, y) 함수 등이 있다

arr1 = [2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]

new_arr1 = np.unique(arr1)
print(new_arr1) # 
print(new_arr1) # [2 5 7 9]

print(np.union1d(new_arr1, arr2))     # [1 2 3 5 7 8 9]
print(np.intersect1d(new_arr1, arr2)) # [2 5]
print(np.setdiff1d(new_arr1, arr2))   # [7 9]

# 7.2.10 난수
# NumPy는 np.random 모듈로 난수를 생성할 수 있다
# * 난수 : 정의된 범위 내에서 무작위로 추출된 수
# 대표적인 함수인 np.random()은 0 이상 1 미만의 난수를 생성하는 np.random.rand() 함수,
# x 이상 y 미만의 정수를 z개 생성하는 np.random.randint(x, y, z) 함수,
# 가우스 분포를 따르는 난수를 생성하는 np.random.normal() 함수 등이 있다.

# np.random.rand() 함수는 ()에 넣은 정수의 횟수만큼 난수가 생성된다
# np.random.randint(x, y, z) 함수는 x 이상 y 미만의 정수를 생성하는 점에 주의
# 또한 z에는 (2, 3) 등의 인수를 넣을 수도 있고, 이렇게 하면 2x3 행렬을 생성할 수 있다

# 일반적으로 np.random.randint()와 같이 기술하여 난수를 생성하는 경우가 많지만
# 매번 np.random 을 입력하는 것은 번거롭다
# from numpy.random import randint와 같이 기술해두면 randint()만으로 함수를 사용할 수 있다
# 'from 모듈명 import 모듈내_함수명' 으로 일반화해서 사용할 수 있다

import numpy as np
from numpy.random import randint
arr1 = randint(0, 11, (5, 2))
print(arr1)

'''
[[ 1  9]
 [ 9  8]
 [ 5  0]
 [ 9 10]
 [10  7]]
'''

arr2 = np.random.rand(3)
print(arr2)
# [0.39086882 0.30928709 0.20971149]