# 연습 문제
# 와인의 데이터셋을 사용해서 데이터 클렌징의 기본을 복습

import pandas as pd
import numpy as np
from numpy import nan as NA

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
              "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
              "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines",
              "Proline" ]
            
df_ten = df.head(10)
print(df_ten)
print()


# 데이터 일부 누락
df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print("========데이터 일부 누락========")
print(df_ten)
print()

# fillna() 메서드로 NaN 부분에 열의 평균값을 대입
df_ten.fillna(df_ten.mean())
print("========fillna 평균값========")
print(df_ten)
print()

# "Alcohol" 열의 평균을 출력
print("Alcohol 열의 평균 : ", df_ten["Alcohol"].mean())
print()

# 중복된 행 제거
df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
print("========")
print(df_ten)


df_ten = df_ten.drop_duplicates()
print(df_ten)

# Alcohol 열의 구간 리스트를 작성
alcohol_bins = [0, 5, 10, 15, 20, 25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"], alcohol_bins)

# 구간 수 집계, 출력
print(pd.value_counts(alcoholr_cut_data))