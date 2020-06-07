# CHAPTER 9 : Pandas 응용

# 9.1 DataFrame 연결과 결합의 개요

# Pandas의 DataFrame을 연결하거나 결합할 수 있다
# DataFrame을 일정 방향으로 붙이는 작업을 '연결'이라고 하고
# 특정 key를 참조하여 연결하는 조작을 '결합'이라고 한다

# 9.2 DataFrame 연결
# 9.2.1 인덱스나 컬럼이 일치하는 DataFrame 간의 연결
# DataFrame 을 일정 방향으로 붙이는 작업을 연결이라고 한다
# 먼저 인덱스 또는 컬럼이 일치하는 DatFrame 연결을 살펴보자
# pandas.concat("DatFrame 리스트", axis = 0) 으로 리스트의 선두부터 순서대로 세로로 연결한다
# axis = 1로 지정하면 가로로 연결된다
# 세로 방향으로 연결할 때는 동일한 컬럼으로 연결되며
# 가로 방향으로 연결할 때는 동일한 인덱스로 연결된다
# 그대로 연결하므로 컬럼에 중복된 값이 생길 수 있다

# 문제
import numpy as np
import pandas as pd

# 지정한 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수
def make_random_df(index, columns, seed) :
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns :
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

# 인덱스와 컬럼이 일치하는 DataFrame 만들기
columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

print(df_data1)
print()
print(df_data2)

df1 = pd.concat([df_data1, df_data2], axis = 0) # 세로로 연결
print(df1)
print()
df2 = pd.concat([df_data1, df_data2], axis = 1) # 가로로 연결
print(df2)
print()
'''
   apple  orange  banana
1     45      68      37
2     48      10      88
3     65      84      71
4     68      22      89
1     38      76      17
2     13       6       2
3     73      80      77
4     10      65      72

   apple  orange  banana  apple  orange  banana
1     45      68      37     38      76      17
2     48      10      88     13       6       2
3     65      84      71     73      80      77
4     68      22      89     10      65      72
'''

# 9.2.2 인덱스나 컬럼이 일치하지 않는 DataFrame 간의 연결
# 인덱스나 컬럼이 일치하지 않는 DataFrame끼리 연결하는 경우,
# 공통의 인덱스나 컬럼이 아닌 경우 행과 열에는 NaN 셀이 생성된다
# 'pandas.concat("DataFrame 리스트", axis = 0)'으로 리스트의 선두부터 순서대로 세로로 연결
# axis = 1 을 지정하면 가로로 연결

# 문제
# 인덱스, 컬럼 일치하지 않는 DataFrame 끼리 연결했을 때 어떻게 동작하는지 확인

def make_random_df2(index, columns, seed) :
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns :
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns1 = ["apple", "orange", "banana"]
columns2 = ["orange", "kiwifruit", "banana"]

# 인덱스가 1, 2, 3, 4이고 컬럼이 columns1인 DataFrame을 만듦
df_data1 = make_random_df2(range(1, 5), columns1, 0)

# 인덱스가 1, 3, 5, 7이고 컬럼이 columns2인 DataFrame을 만듦
df_data2 = make_random_df2(np.arange(1, 8, 2), columns2, 1) 

# df_data1과 df_data2를 세로로 연결, df1 에 대입
df1 = pd.concat([df_data1, df_data2], axis = 0)

# df_data1과 df_data2를 가로로 연결, df2 에 대입

df2 = pd.concat([df_data1, df_data2], axis = 1)

print("df1")
print(df1)
print()
print("df2")
print(df2)
print()

# 연결은 이루어짐! 대신 안 맞는 부분들에는 NaN으로 채워진다
'''
df1
   apple  orange  banana  kiwifruit
1   45.0      68      37        NaN
2   48.0      10      88        NaN
3   65.0      84      71        NaN
4   68.0      22      89        NaN
1    NaN      38      17       76.0
3    NaN      13       2        6.0
5    NaN      73      77       80.0
7    NaN      10      72       65.0

df2
   apple  orange  banana  orange  kiwifruit  banana
1   45.0    68.0    37.0    38.0       76.0    17.0
2   48.0    10.0    88.0     NaN        NaN     NaN
3   65.0    84.0    71.0    13.0        6.0     2.0
4   68.0    22.0    89.0     NaN        NaN     NaN
5    NaN     NaN     NaN    73.0       80.0    77.0
7    NaN     NaN     NaN    10.0       65.0    72.0
'''

# 9.2.3 연결 시 라벨 지정하기
# DataFrame끼리 연결하면 라벨이 중복되는 경우가 있다
# 이 경우 pd.concat()에 keys를 추가하여 라벨 중복을 피할 수 있다
# 연결한 뒤의 DataFrame은 복수 라벨이 사용된 MultiIndex가 된다

# 문제
def make_random_df3(index, columns, seed) :
    np.random.seed(seed) 
    df = pd.DataFrame()
    for column in columns :
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

# df_data1과 df_data2를 가로로 연결하고 key로 "X", "Y"를 지정하여 MultiIndex로 만든 뒤 df에 대입
df = pd.concat([df_data1, df_data2], axis = 1, keys = ["X", "Y"])

print(df)
print()

# df의 "Y" 라벨 "banana"를 Y_banana에 대입
Y_banana = df["Y", "banana"]
print(Y_banana)
print()

'''
      X                   Y
  apple orange banana apple orange banana
1    45     68     37    38     76     17
2    48     10     88    13      6      2
3    65     84     71    73     80     77
4    68     22     89    10     65     72

1    17
2     2
3    77
4    72
Name: (Y, banana), dtype: int32

'''