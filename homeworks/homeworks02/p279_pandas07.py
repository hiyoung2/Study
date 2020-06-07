# 9.4 DataFrame을 이용한 데이터 분석

# 9.4.1 특정 행 얻기
# Pandas에서 데이터양이 방대할 때 화면에 전부 출력하는 것은 어렵다
# DataFrame 형의 변수 df에 대해 df.head()는 첫 5행만 담긴 DataFrame을 반환한다
# 마찬가지로 df.tail()은 끝 5행만 담긴 DataFrame을 반환한다
# 또한 인수로 정숫값을 저장하면 처음이나 끝에서부터 특정 행까지의 DataFrame을 얻을 수 있다
# head() 메서드와 tail() 메서드에 Series형 변수를 사용할 수도 있다

# 문제

import numpy as np
import pandas as pd

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가
df = pd.DataFrame()
for column in columns :
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

print(df)
print()
'''
    apple  orange  banana  strawberry  kiwifruit
1       6       8       6           3         10
2       1       7      10           4         10
3       4       9       9           9          1
4       4       9      10           2          5
5       8       2       5           4          8
6      10       7       4           4          4
7       4       8       1           4          3
8       6       8       4           8          8
9       3       9       6           1          3
10      5       2       1           2          1
'''
df_head = df.head(3)

df_tail = df.tail(3)

print("df_head")
print(df_head)
print()
print("df_tail")
print(df_tail)
print()

'''
df_head
   apple  orange  banana  strawberry  kiwifruit
1      6       8       6           3         10
2      1       7      10           4         10
3      4       9       9           9          1

df_tail
    apple  orange  banana  strawberry  kiwifruit
8       6       8       4           8          8
9       3       9       6           1          3
10      5       2       1           2          1
'''

# 9.4.2 계산 처리하기
# Pandas와 NumPy는 상호호환이 좋아서 유연한 데이터 전달이 가능하다
# NumPy에서 제공하는 함수에 Series나 DataFrame을 전다하여 전체 요소를 계산할 수 있다
# Numpy 배열을 받아들이는 함수에 DataFrame을 전달하는 경우 열 단위로 정리하여 계산된다

# 또한 Pandas는 NumPy처럼 브로드캐스트를 지원하므로 Pandas 간의 계산 혹은 Pandas와 정수 간의 계산을
# +, -, *, /를 사용해서 유연하게 처리할 수 있다

# 문제

import numpy as np
import pandas as pd
import math

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가
df = pd.DataFrame()
for column in columns :
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 각 요소를 두 배로 만들어 double_df에 대입
double_df = df * 2

# df의 각 요소를 제곱하여 square_df에 대입
square_df = df * df

# df의 각 요소를 제곱근 계산하여 sqrt_df에 대입
sqrt_df = np.sqrt(df)

print("double")
print(double_df)
print()
print("square")
print(square_df)
print()
print("sqrt")
print(sqrt_df)
print()

'''
double
    apple  orange  banana  strawberry  kiwifruit
1      12      16      12           6         20
2       2      14      20           8         20
3       8      18      18          18          2
4       8      18      20           4         10
5      16       4      10           8         16
6      20      14       8           8          8
7       8      16       2           8          6
8      12      16       8          16         16
9       6      18      12           2          6
10     10       4       2           4          2

square
    apple  orange  banana  strawberry  kiwifruit
1      36      64      36           9        100
2       1      49     100          16        100
3      16      81      81          81          1
4      16      81     100           4         25
5      64       4      25          16         64
6     100      49      16          16         16
7      16      64       1          16          9
8      36      64      16          64         64
9       9      81      36           1          9
10     25       4       1           4          1

sqrt
       apple    orange    banana  strawberry  kiwifruit
1   2.449490  2.828427  2.449490    1.732051   3.162278
2   1.000000  2.645751  3.162278    2.000000   3.162278
3   2.000000  3.000000  3.000000    3.000000   1.000000
4   2.000000  3.000000  3.162278    1.414214   2.236068
5   2.828427  1.414214  2.236068    2.000000   2.828427
6   3.162278  2.645751  2.000000    2.000000   2.000000
7   2.000000  2.828427  1.000000    2.000000   1.732051
8   2.449490  2.828427  2.000000    2.828427   2.828427
9   1.732051  3.000000  2.449490    1.000000   1.732051
10  2.236068  1.414214  1.000000    1.414214   1.000000
'''



# 9.4.3 통계 정보 얻기
# 컬럼별로 데이터의 평균값, 최댓값, 최솟값 등의 통계 정보를 집계할 수 있다
# DataFrame형 변수 df를 df.describe() 하여 컬럼당 데이터 수, 평균값, 표준편차, 최솟값, 사분위수(25%, 50%, 75%), 최댓값 정보를
# 포함하는 DataFrame을 반환한다
# 반환된 DataFrame의 인덱스는 통계 정보의 이름이 된다

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame 을 생성하고 열을 추가
df = pd.DataFrame()
for column in columns :
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

print("df")
print(df)
print()

# df의 통계 정보 중 "mean", "max", "min"을 꺼내서 df_dbs에 대입

df_dbs = df.describe().loc[["mean", "max", "min"]]

print("df_dbs")
print(df_dbs)
print()

'''
df_dbs
      apple  orange  banana  strawberry  kiwifruit
mean    5.1     6.9     5.6         4.1        5.3
max    10.0     9.0    10.0         9.0       10.0
min     1.0     2.0     1.0         1.0        1.0
'''

# 9.4.4 DataFrame의 행간 차이와 열간 차이 구하기
# 행간 차이를 구하는 작업은 시계열 분석(time-series analysis) 에 자주 이용된다
# DataFrame형 변수 df에 대해 'df.diff("행 간격 또는 열 간격", axis = "방향")을 지정하면 
# 행간 또는 열간 차이를 계산하는 DataFrame이 생성된다
# 첫 번째 인수가 양수이면 이전 행과의 차이를, 음수면 다음 행과의 차이를 구한다
# axis = 0인 경우에는 행의 방향, 1인 경우에는 열의 방향을 가리킨다

# 문제
np.random.seed(0)

columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가한다
df = pd.DataFrame()
for column in columns :
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# df의 각 행에 대해 2행 뒤와의 차이를 계산한 DataFrame을 df_diff에 대입

df_diff = df.diff(-2, axis = 0)

print("df")
print(df)
print()
print("df_diff")
print(df_diff)
print()

'''
df
    apple  orange  banana  strawberry  kiwifruit
1       6       8       6           3         10
2       1       7      10           4         10
3       4       9       9           9          1
4       4       9      10           2          5
5       8       2       5           4          8
6      10       7       4           4          4
7       4       8       1           4          3
8       6       8       4           8          8
9       3       9       6           1          3
10      5       2       1           2          1

df_diff
    apple  orange  banana  strawberry  kiwifruit
1     2.0    -1.0    -3.0        -6.0        9.0
2    -3.0    -2.0     0.0         2.0        5.0
3    -4.0     7.0     4.0         5.0       -7.0
4    -6.0     2.0     6.0        -2.0        1.0
5     4.0    -6.0     4.0         0.0        5.0
6     4.0    -1.0     0.0        -4.0       -4.0
7     1.0    -1.0    -5.0         3.0        0.0
8     1.0     6.0     3.0         6.0        7.0
9     NaN     NaN     NaN         NaN        NaN
'''

# 9.4.5 그룹화
# 데이터베이스나 DataFrame의 특정 열에서 동일한 값의 행을 집계하는 것을 그룹화라고 한다
# DataFrame 변수 df에 대해 'df.groupby("컬럼")'으로 지정한 컬럼을 그룹화 할 수 있다
# 이 경우 GroupBy 객체는 반환하지만, 그룹화된 결과는 표시하지 않는다(그룹화만 했을 뿐이다!)
# 그룹화의 결과를 표시하려면 GroupBy 객체에 대해 그룹의 평균을 구하는 mena(), 합을 구하는 sum() 등의 통계함수를 사용한다

# 문제

import pandas as pd
# 도시 정보를 가진 DataFrame을 만든다
perfecture_df = pd.DataFrame([["강릉", 1040, 213527, "강원도"], 
                             ["광주", 430, 1458915, "전라도"],
                             ["평창", 1463, 42218, "강원도"], 
                             ["대전", 539, 1476955, "충청도"],
                             ["단양", 780, 29816, "충청도"]],
                             columns = ["Perfecture", "Area", "Population", "Region"])

print(perfecture_df)
'''
  Perfecture  Area  Population Region
0         강릉  1040      213527    강원도
1         광주   430     1458915    전라도
2         평창  1463       42218    강원도
3         대전   539     1476955    충청도
4         단양   780       29816    충청도
'''

# perfecture_df를 지역(Region)으로 그룹화하여 grouped_region에 대입
grouped_region = perfecture_df.groupby("Region")

# perfecture_df의 지역별 면적(Area)과 인구(Population)의 평균을 mean_df에 대입
mean_df = grouped_region.mean()

print("mean_df")
print(mean_df)
print()
'''
          Area  Population
Region
강원도     1251.5    127872.5
전라도      430.0   1458915.0
충청도      659.5    753385.5
'''