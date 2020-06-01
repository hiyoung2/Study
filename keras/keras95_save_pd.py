# pandas 파일 data 저장하기
# 판다스를 쓰는 이유?
# 데이터 보니까 (iris) 맨 첫 행에(헤더부분) 숫자가 아님, 얘까지 넘파이로 땡겨오면 에러가 난다
# 넘파이로 하려면 헤더는 없애고 가져와야 한다
# 판다스로 땡겨오면 에러 나지 않음

import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", 
                        index_col=None,
                        header=0, sep=',')
                        # 판다스에서는 데이터 간격을 콤마로 나누는데 여기서 콤마를 빼 준다
# index_col = None : index column은 들어 있지 않다
# hearder = 0, 첫 번째 행(?) header로 처리됨
# header = None 해 버리면
#          0    1       2           3          4
# 0    150.0  4.0  setosa  versicolor  virginica
# 이렇게 됨, 150.0 ~~~ 이거 자체는 데이터가 아님!
# 따라서 header = 0, 첫째 행은 데이터로 인식하지 말라고 지정해준다
# header = 0 으로 지정해주면
#   150    4  setosa  versicolor  virginica
# 이렇게 실제 데이터에 포함 되지 않고 header로 따로 분리된다
print(datasets)
# 자동으로 index 생성됨

print(datasets.head()) # 위에서부터 5개만 보여준다
# [150 rows x 5 columns]
#    150    4  setosa  versicolor  virginica
# 0  5.1  3.5     1.4         0.2          0
# 1  4.9  3.0     1.4         0.2          0
# 2  4.7  3.2     1.3         0.2          0
# 3  4.6  3.1     1.5         0.2          0
# 4  5.0  3.6     1.4         0.2          0


print(datasets.tail()) # 아래서부터 5개를 보여준다
#      150    4  setosa  versicolor  virginica
# 145  6.7  3.0     5.2         2.3          2
# 146  6.3  2.5     5.0         1.9          2
# 147  6.5  3.0     5.2         2.0          2
# 148  6.2  3.4     5.4         2.3          2
# 149  5.9  3.0     5.1         1.8          2

# 기타 자세한 사항은 알아서 공부하자

# index와 header가 있다는 것을 잊지 말자

print("=============================")
print(datasets.values) # 머신을 돌리기 위해서 pandas를 numpy로 바꿔준다!!!!!!! 항상 쓰게 될 것! 
                       # 배열의 형태가 등장함, 친숙한 친구 ㅋㅋ
                       # 수치로만 되어 있는 데이터를 볼 수 있다!

aaa = datasets.values
print(type(aaa))  # <class 'numpy.ndarray'>


np.save('./data/iris.npy', arr=aaa)