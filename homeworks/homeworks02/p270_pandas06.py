# 9.3 DataFrame 결합
# 9.3.1 결합 유형

# '결합'은 '병합(merge)' 라고도 부른다
# 결합은 Key로 불리는 열을 지정하고, 두 데이터베이스의 Key 값이 일치하는 행을 옆으로 연결한다
# 결합은 크게 내부결합과 외부결합 두 가지 방법이 있다

# 내부 결합
# Key 열이 공통되지 않는 행은 삭제된다
# 또한 동일한 컬럼이지만 값이 일치하지 않는 행의 경우 이를 남기거나 없앨 수 있다

# 외부 결합
# Key 열이 공통되지 않아도 행이 삭제되지 않고 남는다
# 공통되지 않은 열에는 NaN 셀이 생성된다

# 9.3.2 내부 결합의 기본
# df1, df2 두 DataFrame 에 대해 'pandas.merge(df1, df2, on=Key가될컬럼, how = "inner")'로 
# 내부 결합된 DataFrame을 생성할 수 있다, 이 경우 df1이 왼쪽에 위치한다

# Key열의 값이 일치하지 않는 행은 삭제된다
# Key가 아니면서 이름이 같은 열은 접미사가 붙는다
# 왼쪽 DataFrame의 열에는 _x가, 오른쪽 DataFrame의 컬럼에는 _y가 접미사로 붙는다
# 사용자가 지정하지 않는 한 DataFrame의 인덱스는 처리에 관여하지 않는다

# 문제
import numpy as np
import pandas as pd

data1 = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year" : [2001, 2002, 2001, 2008, 2006],
         "amount" : [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits" : ["apple", "orange", "banana", "strawberry", "mango"],
         "year" : [2001, 2002, 2001, 2008, 200],
         "price" : [150, 120, 100, 250, 3000]}

df2 = pd.DataFrame(data2)

print("df1")
print(df1)
print()
print("df2")
print(df2)
print()

'''
df1
       frutis  year  amount
0       apple  2001       1
1      orange  2002       4
2      banana  2001       5
3  strawberry  2008       6
4   kiwifruit  2006       3

df2
       frutis  year  price
0       apple  2001    150
1      orange  2002    120
2      banana  2001    100
3  strawberry  2008    250
4       mango   200   3000
'''

# df1, df2 의 컬럼 "fruits"를 Key로 하여 내부 결합 -> df3 에 대입

df3 = pd.merge(df1, df2, on="fruits", how = "inner")
print("df3")
print(df3)
print()
'''
df3
       fruits  year_x  amount  year_y  price
0       apple    2001       1    2001    150
1      orange    2002       4    2002    120
2      banana    2001       5    2001    100
3  strawberry    2008       6    2008    250
'''



# 9.3.3 외부 결합의 기본
# df1, df2 두 DataFrame에 대해 'pandas.merge(df1, df2, on = Key가될컬럼, how = "outer")'로 
# 외부 결합된 DataFrame을 생성할 수 있다, 이 경우 df1은 왼쪽에 위치한다

# Key 열의 값이 일치하지 않는 행이 삭제되지 않고 남겨져 NaN으로 채워진 열이 생성된다
# Key가 아니면서 이름이 같은 열은 접미사가 붙는다
# 왼쪽 DataFrame의 컬럼에는 _x가, 오른쪽 DataFrame 의 컬럼에는 _y가 접미사로 붙는다

import numpy as np
import pandas as pd

data1 = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year" : [2001, 2002, 2001, 2008, 2006],
         "amount" : [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits" : ["apple", "orange", "banana" , "strawberry", "mango"],
         "year" : [2001, 2002, 2001, 2008, 2007],
         "price" : [150, 120, 100, 250, 3000]}

df2 = pd.DataFrame(data2)

print("df1")
print(df1)
print()
print("df2")
print(df2)
print()

'''
df1
       fruits  year  amount
0       apple  2001       1
1      orange  2002       4
2       banan  2001       5
3  strawberry  2008       6
4   kiwifruit  2006       3

df2
      fruits  year  price
0      apple  2001    150
1     orange  2002    120
2      banan  2001    100
3  strawbery  2008    250
4      mango  2007   3000

'''

# df1과 df2의 컬럼 "fruits"를 Key로 하여 외부 결합한 DataFrame df3에 대입

df3 = pd.merge(df1, df2, on = "fruits", how = "outer")

print("df3")
print(df3)
print()
'''
df3
       fruits  year_x  amount  year_y   price
0       apple  2001.0     1.0  2001.0   150.0
1      orange  2002.0     4.0  2002.0   120.0
2      banana  2001.0     5.0  2001.0   100.0
3  strawberry  2008.0     6.0  2008.0   250.0
4   kiwifruit  2006.0     3.0     NaN     NaN
5       mango     NaN     NaN  2007.0  3000.0
'''

# 9.3.4 이름이 다른 열을 Key로 결합하기
# 두 개의 DataFrame이  있다
# 왼쪽은 주문 정보를 저장한 order_df이며,, 
# 오른쪽은 고객 정보를 저장한 customer_df이다
# 주문 정보에서는 고객의 ID 컬럼을 "customer_id"라고 한 반면
# 고객 정보에서는 고객의 ID 컬럼을 "id"라고 했다
# 주문 정보에 고객 정보의 데이터를 넣고 싶으므로 "customer_id"를 Key로 하고 싶지만
# customer_df에 대응하는 컬럼이 "id"이므로 대응하는 컬럼명이 일치하지 않는다
# 이런 경우에는 각 컬럼을 별도로 지정해야 한다

# 'pandas.merge(왼쪽 DF, 오른쪽 DF, left_on="왼쪽Df의 컬럼", right_on = "오른쪽 DF의 컬럼", how = "결합방식")'
# 으로 컬럼이 다른 DataFrame 사이의 열을 결합할 수 있다


# 문제
import pandas as pd

# 주문 정보
order_df = pd.DataFrame([[1000, 2546, 103],
                        [1001, 4352, 101],
                        [1002, 342, 101]],
                        columns = ["id", "item_id", "customer_id"])
        
# 고객 정보
customer_df = pd.DataFrame([[101, "광수"],
                           [102, "민호"],
                           [103, "소희"]],
                           columns = ["id", "name"])

order_df = pd.merge(order_df, customer_df, left_on = "customer_id", right_on = "id", how = "inner")


print(order_df)
print()

'''
   id_x  item_id  customer_id  id_y name
0  1000     2546          103   103   소희
1  1001     4352          101   101   광수
2  1002      342          101   101   광수
'''

# 9.3.5 인덱스를 Key로 결합하기
# DataFrame 간의 결합에 사용하는 Key가 인덱스인 경우에는
# 'lef_on = "왼쪽 df 컬럼", right_on = "오른쪽 df 컬럼"' eotls
# 'left_index = True, right_inidex = True'로 지정한다

# 문제

# 주문 정보
order_df = pd.DataFrame([[1000, 2546, 103],
                        [1001, 4352, 101],
                        [1002, 342, 101]],
                        columns = ["id", "item_id", "customer_id"])

# 고객 정보
customer_df = pd.DataFrame([["광수"],
                           ["민호"],
                           ["소희"]],
                           columns = ["name"])
customer_df.index = [101, 102, 103]

# customer_dr를 바탕으로 "name"을 order_df와 결합하여 order_df에 대입

order_df = pd.merge(order_df, customer_df, left_on = "customer_id", right_index = True, how = "inner")

print(order_df)
print()

'''
     id  item_id  customer_id name
0  1000     2546          103   소희
1  1001     4352          101   광수
2  1002      342          101   광수

'''

