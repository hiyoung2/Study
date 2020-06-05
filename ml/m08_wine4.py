# 등급 수가 너무 다양 -> 줄여보자
# 시험, 대회 같은 곳에서는 y data 자체를 제공을 잘 해 주지만(실제로 y data를 건드릴 일이 없다)
# 실전 상황, 업무 상황에서는 y data를 잘 판단해서 레이블 축소 등의 조절을 해 줘야 한다


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep = ';', header = 0)

y = wine['quality']
x = wine.drop('quality', axis = 1) # x는 wine data에서 quality 열을 drop하겠다!

print('x.shape : ', x.shape) # x.shape :  (4898, 11)
print('y.shape : ', y.shape) # y.shape :  (4898,)

# y 레이블 축소
newlist = []

for i in list(y) :
    if i <= 4 :
        newlist +=[0]
    elif i <= 7 :
        newlist +=[1]
    else : 
        newlist +=[2]

y = newlist

# 3, 4 / 5, 6, 7 / 8, 9 / 이렇게 3등급으로 축소

print('y_newlist : ', y)
# 0, 1, 2 -> 3등급


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8
)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_pred))
print("acc       : ", acc)

# acc_socre :  0.9326530612244898
# acc       :  0.9326530612244898