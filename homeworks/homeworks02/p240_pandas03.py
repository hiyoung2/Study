# 8.3 DataFrame 
# 8.3.1 DataFrame 생성

# DataFrame은 Series를  여러 개 묶은 것 같은 2차원 데이터 구조를 하고 있다
# DataFrame은 pd.DataFrame()에 Series를 전달하여 생성할 수 있다
# 행에는 0부터 오름차순으로 번호가 붙어 있다!!!!!
# pd.DataFrame([Series, Series, ...])

# DataFrame의 값으로 딕셔너리형(리스트 포함)을 넣어도 된다
# 참고로 해당 리스트형의 길이는 동일해야 한다

import pandas as pd

data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruits"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(df)
print()
'''
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4  kiwifruits  2006     3
'''

# 문제
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]

series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)

df = pd.DataFrame([series1, series2])
print(df)
print()
'''
   apple  orange  banana  strawberry  kiwifruit
0     10       5       8          12          3
1     30      25      12          10          8
'''

# 8.3.2 인덱스와 컬럼 설정
# DataFrame에서는 행의 이름을 인덱스, 열의 이름을 컬럼이라고 부른다
# 인수를 지정하지 않고 DataFrame을 작성하면 0부터 오름차순으로 인덱스가 할당된다
# 컬럼은 원본 데이터 Series의 idnex 및 딕셔너리형의 Key가 된다
# DataFrame형 변수 df의 인덱스는 df.index 에 행 수와 같은 길이의 리스트를 대입하여 설정할 수 있다
# df의 컬럼은 df.columns에 열 수와 같은 길이의 리스트를 대입하여 설정할 수 있다
# df.index = ["name1", "name2"]