# wine data 자체적으로 문제가 있다
# acc가 잘 나오지 않았음
# y data인 quality 의 값들이 한 쪽으로 치우쳐 있다
# 다른 예제 데이터 iris, mnist는 골고루 나눠져있던 반면에! 그래서 acc가 잘 나왔었음

import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep = ';', header = 0)

# 판다스의 groupby : 컬럼 안의 것들을 그룹별로 모아준다
wine.groupby('quality')['quality'].count()

# count_data라는 변수에 컬럼 안의 갯수들을 볼 수 있는 걸 넣는다
count_data = wine.groupby('quality')['quality'].count()

print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64

5, 6에만 집중 분포
'''

count_data.plot()
plt.show()