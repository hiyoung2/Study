# 연습 문제
# df1와 df2는 각각 야채, 과일에 대한 DataFrame이다
# "Name", "Type", "Price"는 각각 이름, 종류(야채? 과일?), 가격을 나타낸다
# 야채와 과일을 각각 3개씩 구입할 것, 저렴하게 구매하기 위해 다음 순서로 최소 비용을 구하라
# - df1와 df2는 세로로 결합
# - 야채와 과일을 각각 추출하여 "Price"로 정렬
# - 야채와 과일을 저렴한 순으로 위에서 세 개씩을 선택하여 총액을 계산하고 출력

import pandas as pd

df1 = pd.DataFrame([["apple", "Fruit", 120],
                   ["orange", "Fruit", 60],
                   ["banana", "Fruit", 100],
                   ["pumpkin", "Vegetable", 150],
                   ["potato", "Vegetable", 80]],
                   columns = ["Name", "Type", "Price"])

df2 = pd.DataFrame([["onion", "Vegetable", 60],
                   ["carrot", "Vegetable", 50],
                   ["beans", "Vegetable", 100],
                   ["grape", "Fruit", 160],
                   ["kiwifruit", "Fruit", 80]],
                   columns = ["Name", "Type", "Price"])

df3 = pd.concat([df1, df2], axis = 0)

# 야채 추출, Price로 정렬
df_vege = df3.loc[df3["Type"] == "Vegetable"]
df_vege = df_vege.sort_values(by = "Price")

# 과일 추출, Price로 정렬
df_fruit = df3.loc[df3["Type"] == "Fruit"]
df_fruit = df_fruit.sort_values(by = "Price")

print("df_vege")
print(df_vege)
print()
print("df_fruit")
print(df_fruit)
print()

'''
df_vege
      Name       Type  Price
1   carrot  Vegetable     50
0    onion  Vegetable     60
4   potato  Vegetable     80
2    beans  Vegetable    100
3  pumpkin  Vegetable    150

df_fruit
        Name   Type  Price
1     orange  Fruit     60
4  kiwifruit  Fruit     80
2     banana  Fruit    100
0      apple  Fruit    120
3      grape  Fruit    160
'''

# 야채, 과일 저렴한 순 3개, 총합 구하기
df_sum = sum(df_vege[:3]["Price"]) + sum(df_fruit[:3]["Price"])

print("df_sum :", df_sum) # df_sum : 430
