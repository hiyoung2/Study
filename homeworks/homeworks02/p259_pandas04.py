# 연습문제

import pandas as pd
import numpy as np

index = ["growth", "mission", "ishikawa", "pro"]
data = [50, 7, 26, 1]

# Series 작성
series = pd.Series(data, index = index)
print("series")
print(series)
print()

# 인덱스를 알파벳순으로 정렬한 series를 aidemy에 대입
aidemy = series.sort_index(ascending = True)
print("aidemy")
print(aidemy)
print()

# 인덱스가 "tutor" 이고 데이터가 30인 요소를 series에 추가
aidemy1 = pd.Series([30], index = ["tutor"])
aidemy2 = series.append(aidemy1)

print("aidemy2")
print(aidemy2)
print()

# DataFrame을 생성하고 열을 추가
df = pd.DataFrame()
for index in index :
    df[index] = np.random.choice(range(1, 11), 10)

df.index = range(1, 11)

aidemy3 = df.loc[range(2, 6), ["ishikawa"]]
print()
print(aidemy3)