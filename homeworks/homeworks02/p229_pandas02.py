# 8.2 Series
# 8.2.1 Series 생성

# Pandas의 데이터 구조 중 하나인 Series는 1차원 배열처럼 다룰 수 있다
# Pandas를 import 한 뒤 'pandas.Series(딕셔너리형의_리스트)'로 딕셔너리형 리스트를 전달해서 Series를 생성할 수 있다
# 또한 import pandas as pd라고 입력하면 pandas.Series를 pd.Series로 줄여서 적을 수 있다(import numpy as np와 같음)

# 데이터와 관련된 인덱스를 지정해도 Series를 생성할 수 있다
# 'pd.Series(데이터_배열, index = 인덱스_배열)'로 Series를 생성할 수 있다
# 인덱스를 지정하지 않으면 0부터 순서대로 정수 인덱스가 붙는다
# Series를 출력하면 'dtype : int64' 라고 출력된다
# 이는 Seires에 저장되어 있는 값이 in64형임을 보여준다
# dtype은 data type으로, 데이터의 자료형을 나타낸다(데이터가 정수면 int, 소수점이 있으면 float 등)
# int64는 64bit의 크기를 가진 정수로 -2^63 ~ 2^63-1 까지의 정수를 처리할 수 있다
# 그 밖에도 dtype에는 int32 등과 같이 같은 정수형이더라도 크기가 다른 것, 0 또는 1 값만 가지는 bool 형 등이 있다

import pandas as pd

fruits = {"banana" : 3, "orange" : 2}
print(pd.Series(fruits))
'''
banana    3
orange    2
dtype: int64
'''

print()

# 문제
index = ["apple", "orange", "banan", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

print(series)
'''
apple         10
orange         5
banan          8
strawberry    12
kiwifruit      3
dtype: int64
'''

print()

# 8.2.2 참조
# Series의 요소를 참조할 때는 번호를 지정하거나 인덱스값을 지정하는 방법을 사용할 수 있다
# 번호를 지정하는 경우 리스트의 슬라이스 기능처럼 series[:3] 등으로 지정하여 원하는 범위를 추출할 수 있다
# 인덱스값을 지정하는 경우 원하는 요소의 인덱스값을 하나의 리스트로 정리한 뒤 참조할 수 있다
# 리스트 대신 하나의 정숫값을 지정하여 그 위치에 해당하는 데이터만을 추출할 수도 있다

import pandas as pd
fruits = {"banana" : 3, "orange" : 4, "grape" : 1, "peach" : 5}
series = pd.Series(fruits)
print(series[0:2])
'''
banana    3  sereis [0]에 해당
orange    4  series [1]에 해당 / [2] 끝 숫자는 포함 안 됨!
dtype: int64
'''
print()
print(series[["orange", "peach"]])
'''
orange    4
peach     5
dtype: int64
# 지정한 인덱스들만 출력된다
'''

print()

# 문제 
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)
print(series)
print()

# 인덱스 참조를 사용, series의 2 ~ 4 번째 세 요소를 추출, items1에 대입
items1 = series[1:4]

# 인덱스 값을 지정하는 방법으로 "apple", "banana", "kiwifruit" 의 인덱스를 가진 요소를 추출, items2에 대입
items2 = series[["apple", "banana", "kiwifruit"]]


print(items1)
print()
print(items2)
print()
'''
orange         5
banan          8
strawberry    12
dtype: int64

apple        10
banana        8
kiwifruit     3
dtype: int64
'''

# 8.2.3 데이터와 인덱스 추출
# 작성한 Series의 데이터값 또는 인덱스를 추출하는 방법이 있다
# Series 자료형은 series.values로 데이터 값을 참조할 수 있고, series.index로 인덱스를 참조할 수 있다

# 문제
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

series_values = series.values

series_index = series.index

print(series_values)
print()
print(series_index)
print()
'''
[10  5  8 12  3]

Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')
'''

# 8.2.4 요소 추가
# Series에 요소를 추가하려면 해당 요소도 Series형이어야 한다
# 추가할 요소를 Series형으로 변환한 뒤! Series형의 append()로 전달하여 추가할 수 있다

fruits = {"banana" : 3, "oragne" : 2}
series = pd.Series(fruits)
series = series.append(pd.Series([3], index = ["grape"]))
print(series)
print()
'''
banana    3
oragne    2
grape     3
dtype: int64
'''
# 문제
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

series = series.append(pd.Series([12], index = ["pineapple"]))
print(series)
print()

# 아래와 같은 방식으로 해도 된다
# pineapple = pd.Sereis([12], index = ["pineapple"])
# series = series.append(pineapple) 

'''
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
pineapple     12
dtype: int64
'''

# 8.2.5 요소 삭제
# Series의 인덱스 참조를 사용하여 요소를 삭제할 수 있다
# Series형의 변수 series에서 'series.drop("인덱스")' 를 하여 해당 인덱스 위치의 요소를 삭제할 수 있다

# 문제 
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

# strawberry 요소 삭제 , seires에 대입
series = series.drop("strawberry")
print("series.drop")
print(series)
print()
'''
series.drop
apple        10
orange        5
banana        8
kiwifruit     3
dtype: int64
'''

# 8.2.6 필터링
# Series형 데이터에서 조건과 일치하는 요소를 꺼내고 싶을 때가 있다
# Pandas에서는 bool형의 시퀀스를 지정해서 True인 것만을 추출할 수 있다
# Sequence란 '연속' 또는 '순서'를 말한다

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

conditions = [True, True, False, False, False]
print(series[conditions])
print()
'''
apple     10
orange     5
dtype: int64
# True 인 것들만 출력되었다!
'''

# Pandas 에서는 Series(또는 DataFrame)을 사용해서 조건식을 만들어도 bool형의 시퀀스를 취득할 수 있다
# 예를 들어 Series 변수 series에 대해 series[series >= 5]로 값이 5이상인 요소만 가지는 Series를 취득할수 있다
# 또한, series[][] 처럼 []를 여러 개 덧붙여서 복수의 조건을 추가할 수 있다
print("값이 5 이상인 series")
print(series[series >= 5])
print()
'''
값이 5 이상인 series
apple         10
orange         5
banana         8
strawberry    12
dtype: int64
'''

# 문제 : 5 이상 10 미만 요소 포함 series 
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

print("값이 5 이상 10 미만인 series")
print(series[series >= 5][series < 10]) # []를 덧붙임에 따라 조건을 추가할 수 있다!!!
print()
'''
값이 5 이상 10 미만인 series
orange    5
banana    8
dtype: int64
'''

# 8.2.7 정렬
# Series는 인덱스 정렬과 데이터 정렬 방법이 준비되어 있다
# Seires형 변수 series에서 인덱스 정렬은 series.sort_index()로, 
# 데이터 정렬은 series.sort_values()로 할 수 있다
# 특별히 인수를 지정하지 않으면 오름차순으로 정렬된다
# 인수에 ascending = False를 전달하면 내림차순으로 정렬된다

# 문제

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

# series의 인덱스를 알파벳순으로 정렬 items1에 대입
items1 = series.sort_index()
print(items1)
print()
'''
apple         10
banana         8
kiwifruit      3
orange         5
strawberry    12
dtype: int64
'''

# series의 데이터값을 오름차순으로 정렬 items2에 대입
items2 = series.sort_values()
print(items2)
print()
'''
kiwifruit      3
orange         5
banana         8
apple         10
strawberry    12
dtype: int64
'''

# 내림차순으로
items3 = series.sort_index(ascending = False)
print(items3)
print()
'''
strawberry    12
orange         5
kiwifruit      3
banana         8
apple         10
dtype: int64
'''

items4 = series.sort_values(ascending = False)
print(items4)
'''
strawberry    12
apple         10
banana         8
orange         5
kiwifruit      3
dtype: int64
'''