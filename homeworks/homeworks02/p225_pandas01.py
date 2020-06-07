# CHAPTER 8. Pandas 기초

# 8.1. Pandas 개요
# Pandas는 NumPy와 같이 데이터셋을 다루는 라이브러리이다
# NumPY는 데이터를 수학의 행렬로 처리할 수 있으므로 과학 계산에 특화되어 있다
# Pandas는 일반적인 데이터베이스에서 이뤄지는 작업을 수행할 수 있으며
# 수치뿐 아니라 이름, 주소 등의 문자열 데이터도 쉽게 처리할 수 있다
# NumP와 Pandas를 적절히 구사하면 효율적으로 데이터를 분석할 수 있따

# Pandas에는 Series와 DataFrame 의 두 가지 데이터 구조가 존재한다
# 주로 사용되는 데이터 구조는 2차원 테이블로 나타내는 DataFrame 이다
# 가로 방향의 데이터는 행, 세로 방향의 데이터는 열이라고 한다
# 각 행과 열에는 라벨, label이 부여되어 있으며
# 행의 label은 index
# 열의 label은 column 이라고 부른다

# Series는 1차원 배열로 DataFrame의 '행 또는 열' 로도 볼 수 있다
# Series도 각 요소에 라벨이 붙어 있다

# 8.1.1 seires와 DataFrame의 데이터 확인
# Pandas 에는 Seires와 DataFrame의 두 가지 데이터 구조가 있다고 했다
# 실제로 어떤 데이터 구조일까?
# Series에 딕셔너리형을 전달하면 키(key)에 의해 오름차순으로 정렬된다
# * 딕셔너리 : {key : value} 형태!!

# Series는 라벨이 붙은 1차원 데이터!
# DataFrame은 여러 Series를 묶은 것과 같은 2차원 데이터 구조이디ㅏ

import pandas as pd

# Series 데이터
fruits = {"orange" : 2, "banana" : 3}
# print(pd.Series(fruits)) # 이렇게 바로 print 문에 넣어도 되고
# 시리즈 형식으로 만든 것을 fruits에 대입해는 방식으로 해도 결과는 같음
fruits = pd.Series(fruits)
print(fruits)

'''
orange    2
banana    3
dtype: int64
'''

# DataFrame 데이터
data = {"fruits" : ["apple", "orange", "banana", "strawberryr", "kiwifruit"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(df)
'''
        fruits  year  time
0        apple  2001     1
1       orange  2002     4
2       banana  2001     5
3  strawberryr  2008     6
4    kiwifruit  2006     3
index가 자동 생성되었다!
'''
print()

# 문제 : 다음을 실행해서 Series와 DataFrame이 어떤 데이터인지 확인
import pandas as pd

# Series용 라벨(인덱스) 작성
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
# Series용 데이터 대입
data = [10, 5, 8, 12, 3]
# Series 작성
series = pd.Series(data)

# 딕셔너리형을 사용하여 DataFrame용 데이터 작성
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruits"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]}
    
# DataFrame을 작성
df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print()
print("DataFrame 데이터")
print(df)

'''
Series 데이터
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
dtype: int64
# 지금은 인덱스를 지정한 상태, 지정하지 않으면 0부터 오름차순으로 번호가 매겨진다

Series 데이터
0    10
1     5
2     8
3    12
4     3
dtype: int64
# 이런 식으로 0부터 오름차순으로 자동 인덱스가 생성된다

DataFrame 데이터
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4  kiwifruits  2006     3
# DataFrame 행의 인덱스 역시 0부터 오름차순으로 번호가 매겨진다
'''