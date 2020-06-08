# dacon 대회 첫 참가 : 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sep 디폴트 : comma

train = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)

test = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)

submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", train.shape)             # (10000, 75) : x_train, x_test로 만들어야 함(y_train, y_test도 / y가 포함되어 있으니까)
print("test.shape : ", test.shape)               # (10000, 71) : x_pred
print("submission.shape : ", submission.shape)   # (10000, 4)  : y_pred(target 값, 현재 데이터가 없음)

# train.csv : x, y -> x_train, x_test, y_train, y_test
# test.csv : x -> x_pred
# submission.csv : y -> y_pred

# 결측치
print(train.isnull().sum()) # train 데이터 중에 null 값을 sum 해라 
'''
rho           0
650_src       0
660_src       0
670_src       0
680_src       0
           ...
990_dst    1987
hhb           0
hbo2          0
ca            0
na            0
'''
train = train.interpolate() # 보간법 중 '선형보간'이라고 한다 / 완벽하진 않지만 평타 85점
                            # 컬럼별로 선을 그려서 빈 자리를 그 선에 맞게끔 그려서 채워준다
# print(train.isnull().sum()) 
'''
rho        0
650_src    0
660_src    0
670_src    0
680_src    0
          ..
990_dst    0
hhb        0
hbo2       0
ca         0
na         0
'''
# 결측치를 처리하는 방법은 여러 가지 시도해보는 것이 가장 좋다(시간이 있을 때)
# 짧은 시간 동안에 해결해야 한다면, 쌓은 경험치로 직관적? 으로 처리해야 한다(경험치가 중요하네요)

test = test.interpolate()


# csv 파일 만들기(submit 파일)
# y_pred.to_csv(경로)