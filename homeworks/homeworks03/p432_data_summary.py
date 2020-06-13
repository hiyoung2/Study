# 14.4 데이터 요약
# 14.4.1 키별 통계량 산출

# 여기서는 키별 통계량을 산출하겠다
# 와인의 데이터셋을 사용해 열의 평균값 산출

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
              "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
              "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines",
              "Proline" ]

print(df["Alcohol"].mean()) # 13.000617977528083
print()

# 문제 
# 와인의 데이터셋에서 Magnesium 의 평균값 출력
print(df["Magnesium"].mean()) # 99.74157303370787
print()

# 14.4.2 중복 데이터

# 중복 데이터 작성
from pandas import DataFrame

dupli_data = DataFrame({"col1" : [1, 1, 2, 3, 4, 4, 6, 6],
                       "col2" : ["a", "b", "b", "b", "c", "c", "b", "b"]})

print(dupli_data)
print()
'''
   col1 col2
0     1    a
1     1    b
2     2    b
3     3    b
4     4    c
5     4    c
6     6    b
7     6    b
'''

# duplicated() 메서들를 사용하면 중복된 행을 True로 표시한다
# 출력 결과는 지금까지 다룬 DataFrame 형과는 달리 Series 형이며, 외형적인 모습도 다르다

print(dupli_data.duplicated())
'''
0    False
1    False
2    False
3    False
4    False
5     True
6    False
7     True
dtype: bool
'''
# dtype은 'DataType'으로 요소의 자료형을 나타낸다
# drop_duplicates() 메서드를 사용하면 중복 데이터가 삭제된 후의 데이터를 보여줌

print(dupli_data.drop_duplicates())
'''
   col1 col2
0     1    a
1     1    b
2     2    b
3     3    b
4     4    c
6     6    b
'''

# 문제
# 중복 데이터 삭제, 새로운 DataFrame 출력

dupli_data = DataFrame({"col1" : [1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2" : ["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})

print(dupli_data.drop_duplicates())
print()
'''
    col1 col2
0      1    a
1      1    b
2      2    b
3      3    b
4      4    c
6      6    b
8      7    d
10     7    c
11     8    b
12     9    c
'''

# 14.3.3 매핑
# 매핑, mapping은 공통의 키 역할을 하는 데이터의 값을 가져오는 처리이다

import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID" : ["100", "101", "102", "103", "104", "106", "108", 
                      "110", "111", "113"],
               "city" : ["서울", "부산", "대전", "광주", "서울", "서울", "부산",
                         "대전", "광주", "서울"],
               "birth_year" : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 
                              1990, 1995, 1981],
               "name" : ["영이", "순돌", "짱구", "태양", "션", "유리", "현아",
                         "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)
print()

'''
    ID city  birth_year name
0  100   서울        1990   영이
1  101   부산        1989   순돌
2  102   대전        1992   짱구
3  103   광주        1997   태양
4  104   서울        1982    션
5  106   서울        1991   유리
6  108   부산        1988   현아
7  110   대전        1990   태식
8  111   광주        1995   민수
9  113   서울        1981   호식
'''

# 새 딕셔너리 생성
city_map = {"서울" : "서울", 
            "광주" : "전라도", 
            "부산" : "경상도", 
            "대전" : "충청도"}
print(city_map)
print()

# 처음에 준비했던 attri_data_frame1의 city 컬럼을 기반으로 하여
# 대응하는 지역명을 새 컬럼으로 추가한다
# 이것이 매핑처리이다
# 엑셀의 vlookup 과 같은 처리라 생각하면 이해하기 쉽다

# 새로운 column인 region 을 추가, 해당 데이터가 없는 경우 NAN
attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
print(attri_data_frame1)

# 문제
# 다음의 DataFrame에서 city가 서울이나 대전이면 '중부',
# 광주나 부산이면 '남부' 가 되도록 새 열 (MS)을 추가, 결과를 출력

attri_data1 = {"ID" : ["100", "101", "102", "103", "104", "106", "108", 
                      "110", "111", "113"],
               "city" : ["서울", "부산", "대전", "광주", "서울", "서울", "부산",
                         "대전", "광주", "서울"],
               "birth_year" : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 
                              1990, 1995, 1981],
               "name" : ["영이", "순돌", "짱구", "태양", "션", "유리", "현아",
                         "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

MS_map = {"서울" : "중부", 
          "광주" : "남부", 
          "부산" : "남부", 
          "대전" : "중부"}

attri_data_frame1["MS"] = attri_data_frame1["city"].map(MS_map)

print(attri_data_frame1)
print()
'''
    ID city  birth_year name  MS
0  100   서울        1990   영이  중부
1  101   부산        1989   순돌  남부
2  102   대전        1992   짱구  중부
3  103   광주        1997   태양  남부
4  104   서울        1982    션  중부
5  106   서울        1991   유리  중부
6  108   부산        1988   현아  남부
7  110   대전        1990   태식  중부
8  111   광주        1995   민수  남부
9  113   서울        1981   호식  중부
'''




# 14.4.4 구간 분할
# 구간 분할은 이산 범위의 데이터를 분할하여 집계할 경우에 사용하는 편리한 기능
# 구간을 분할할 리스트를 준비하고, pandas의 cut() 함수로 처리한다

attri_data1 = {"ID" : ["100", "101", "102", "103", "104", "106", "108", 
                      "110", "111", "113"],
               "city" : ["서울", "부산", "대전", "광주", "서울", "서울", "부산",
                         "대전", "광주", "서울"],
               "birth_year" : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 
                              1990, 1995, 1981],
               "name" : ["영이", "순돌", "짱구", "태양", "션", "유리", "현아",
                         "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

# 분할 리스트 만들기
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

# 구간 분할을 실시
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)
print()
'''
0    (1985, 1990]
1    (1985, 1990]
2    (1990, 1995]
3    (1995, 2000]
4    (1980, 1985]
5    (1990, 1995]
6    (1985, 1990]
7    (1985, 1990]
8    (1990, 1995]
9    (1980, 1985]
Name: birth_year, dtype: category
Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]
'''
# ()는 값을 포함하지 않고, []는 값을 포함한다
# (1985, 1990] 은 1985년은 포함하지 않고 1990년은 포함한다
# 1985 초과 1990 이하?

# 각 구간의 수를 집계할 때는 value_counts() 메서드를 사용한다
pd.value_counts(birth_year_cut_data)
print(pd.value_counts(birth_year_cut_data))
print()
'''
(1985, 1990]    4
(1990, 1995]    3
(1980, 1985]    2
(1995, 2000]    1
Name: birth_year, dtype: int64
'''

# 각 구간에 이름을 붙일 수도 있다
group_names = ["first1980", "second198", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins, labels = group_names)
pd.value_counts(birth_year_cut_data)
print(pd.value_counts(birth_year_cut_data))
print()
'''
second198     4
first1990     3
first1980     2
second1990    1
Name: birth_year, dtype: int64
'''

# 미리 분할수를 지정하여 분할 할 수 있다
# 이를 이용하면 거의 비슷한 크기의 구간을 만들 수 있다
# CUT() 함수의 두 번째 인수로 분할수를 전달

pd.cut(attri_data_frame1.birth_year, 2)
print(pd.cut(attri_data_frame1.birth_year, 2))
print()
'''

0      (1989.0, 1997.0]
1    (1980.984, 1989.0]
2      (1989.0, 1997.0]
3      (1989.0, 1997.0]
4    (1980.984, 1989.0]
5      (1989.0, 1997.0]
6    (1980.984, 1989.0]
7      (1989.0, 1997.0]
8      (1989.0, 1997.0]
9    (1980.984, 1989.0]
Name: birth_year, dtype: category
Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]]
'''

# 문제 DataFrame의 ID를 두 구간으로 분할해서 출력

attri_data1 = {"ID" : [100, 101, 102, 103, 104, 106, 108, 110, 111, 113],
               "city" : ["서울", "부산", "대전", "광주", "서울", "서울", "부산",
                         "대전", "광주", "서울"],
               "birth_year" : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 
                              1990, 1995, 1981],
               "name" : ["영이", "순돌", "짱구", "태양", "션", "유리", "현아",
                         "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

pd.cut(attri_data_frame1.ID, 2)
print(pd.cut(attri_data_frame1.ID, 2))
print()
'''
0    (99.987, 106.5]
1    (99.987, 106.5]
2    (99.987, 106.5]
3    (99.987, 106.5]
4    (99.987, 106.5]
5    (99.987, 106.5]
6     (106.5, 113.0]
7     (106.5, 113.0]
8     (106.5, 113.0]
9     (106.5, 113.0]
Name: ID, dtype: category
Categories (2, interval[float64]): [(99.987, 106.5] < (106.5, 113.0]]
'''