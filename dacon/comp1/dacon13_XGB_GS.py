import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel

# 1. 데이터 준비
data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape :", data.shape) # (10000, 75)
print("test.shape :", x_pred.shape)   # (10000, 71)  
print("submission.shape :", submission.shape) # (10000, 4)

data = data.interpolate(axis = 0)
x_pred = x_pred.interpolate(axis = 0)

# print(data.isnull().sum()) 
# print(x_pred.isnull().sum()) 

print()

data = data.fillna(data.mean())
x_pred = x_pred.fillna(x_pred.mean())

# print(data.isnull().sum()) 
# print(x_pred.isnull().sum()) 

print()

# 판다스 데이터 프레임 형태에서 슬라이싱하기(iloc 이용)
x = data.iloc[:, :-4]
y = data.iloc[:, -4:]
print("x.shape :", x.shape) # (10000, 71)
print("y.shape :", y.shape) # (10000, 4)

x = x.values
y = y.values
x_pred = x_pred.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

# xgboost 따로 전처리(스케일링) 하지 않아도 된다

# print("x_train.shape :", x_train.shape) # (8000, 71)
# print("x_test.shape :", x_test.shape)   # (2000, 71)
# print("y_train.shape :", y_train.shape) # (8000, 4)
# print("y_test.shape :", y_test.shape)   # (2000, 4)


# 2. 모델 구성
model = MultiOutputRegressor(XGBRegressor())

model.fit(x_train, y_train)

print(model.estimators_)
print(len(model.estimators_))

# regr_multi_Enet.estimators_[0].coef_
# Thanks for the quick answer! For the sake of completeness: in the case of random forest - regr_multi_RF.estimators_[0].feature_importances_

# print(model.estimators_[0].feature_importances_) # 첫 번째 컬럼의 feature-importance
# print()
'''
column : hhb
[0.06693552 0.00139645 0.0030988  0.00319699 0.00323527 0.00383486
 0.00448795 0.00372167 0.00284728 0.00772241 0.0049525  0.00494427
 0.0047055  0.00422412 0.01559567 0.01963057 0.02110042 0.02613407
 0.01587511 0.01774201 0.00849215 0.00845892 0.01025037 0.00861409
 0.00612483 0.00456509 0.00546599 0.00788208 0.00477499 0.00426368
 0.00590977 0.00600268 0.00440518 0.00442949 0.00682768 0.00447816
 0.00705271 0.00337486 0.00550225 0.00485232 0.00350726 0.005868
 0.00526198 0.00516221 0.00518622 0.00446452 0.00724866 0.02827254
 0.05956348 0.03656759 0.07516245 0.03729071 0.08011907 0.04788335
 0.02152706 0.02772677 0.02439542 0.02143695 0.02973414 0.02980439
 0.0055022  0.00531319 0.00609193 0.02041269 0.01187853 0.00599896
 0.00537579 0.00530506 0.00660817 0.00857045 0.00565162]
'''
# print(model.estimators_[1].feature_importances_) # 두 번째 컬럼의 feature_importance
# print()
''' 
column : hbo2
[0.01818308 0.00470051 0.00794586 0.00738977 0.00799535 0.00776739
 0.00904148 0.01018911 0.01215825 0.00878893 0.00980986 0.01358719
 0.01043203 0.01329048 0.01332352 0.01492581 0.01379292 0.01265211
 0.0133403  0.01026374 0.00847029 0.0132606  0.01504718 0.01259543
 0.01221772 0.01115651 0.00966649 0.01372607 0.01024455 0.01446169
 0.012796   0.01407807 0.01425083 0.01592015 0.01653445 0.01552005
 0.01175162 0.01334341 0.012315   0.01727209 0.01695631 0.01207873
 0.01707567 0.01318212 0.01370112 0.02140738 0.01566834 0.01658707
 0.01284612 0.01677121 0.01715927 0.0166776  0.01864368 0.01642952
 0.0178158  0.01374983 0.01967762 0.01849402 0.01703766 0.0171179
 0.01557655 0.01481014 0.01726607 0.0159339  0.01918053 0.01622955
 0.01261576 0.01317812 0.01514268 0.01921765 0.02756609]
'''
# print(model.estimators_[2].feature_importances_) # 세 번째 컬럼의 feature_importance
# print()

''' 
column : ca
[0.02181026 0.00291276 0.01033327 0.00741912 0.01099812 0.00940695
 0.01082391 0.00890496 0.00953417 0.01137053 0.01177299 0.01083992
 0.01281689 0.01277408 0.01022614 0.01120431 0.01319463 0.00913359
 0.00971275 0.00966825 0.01186894 0.01276494 0.00977475 0.01075116
 0.01144653 0.01244453 0.01226761 0.01197028 0.01353836 0.01150638
 0.01748008 0.0200375  0.01683995 0.01929082 0.02040459 0.02444557
 0.0124816  0.01091682 0.01070442 0.01055778 0.01074219 0.01362653
 0.01581821 0.01511137 0.01462549 0.00999383 0.01283645 0.01466785
 0.01020309 0.01523645 0.01278952 0.01247533 0.0119448  0.01229777
 0.01256294 0.01658168 0.01288221 0.01283557 0.01379927 0.01293176
 0.01424054 0.01583609 0.01269758 0.01636988 0.01532419 0.01578053
 0.02439787 0.04446914 0.03560333 0.03191786 0.01305057]
'''
# print(model.estimators_[3].feature_importances_) # 네 번째 컬럼의 feature_importance
# print()
'''
column : na
[0.00780461 0.0038693  0.00663378 0.00997871 0.00831397 0.00921794
 0.01025982 0.01147987 0.00906979 0.01503582 0.0123268  0.01171637
 0.0137765  0.01292373 0.00935693 0.01068481 0.01034911 0.00838525
 0.01042802 0.01377582 0.01478164 0.01235482 0.01214405 0.01391692
 0.01281177 0.01022105 0.01200064 0.01658412 0.01219132 0.01344761
 0.01838665 0.01318817 0.01350885 0.01228161 0.00999649 0.0107217
 0.01282844 0.01154529 0.01221641 0.01375367 0.00874372 0.01213367
 0.01035815 0.01702565 0.02796359 0.02393061 0.01537645 0.01162419
 0.01451046 0.01672141 0.01443945 0.01562544 0.0131462  0.01306556
 0.01579327 0.01504019 0.02080425 0.01608569 0.01652016 0.01500845
 0.02575894 0.02488359 0.03247346 0.02329325 0.0150684  0.02242089
 0.01486934 0.01677327 0.01772664 0.01409608 0.01445149]
'''

for i in range(len(model.estimators_)) :
    threshold = np.sort(model.estimators_[i].feature_importances_)

    for thresh in threshold :
        selection = SelectFromModel(model.estimators_[i], threshold = thresh, prefit = True)

        parameters = {
            "n_estimators" : [10, 100, 20, 200, 30, 300],
            "learning_rate" : [0.01, 0.03, 0.05, 0.07, 0.09],
            "colsample_bytree" : [0.6, 0.7, 0.8, 0.9],
            "colsample_bylevel" : [0.6, 0.7, 0.8, 0.9],
            "max_depth" : [3, 4, 5, 6]
        } 

        search = RandomizedSearchCV(XGBRegressor(), parameters, cv = 5)

        select_x_train = selection.transform(x_train)

        m_search = MultiOutputRegressor(search, n_jobs = -1)
        m_search.fit(select_x_train, y_train)

        select_x_test = selection.transform(x_test)

        y_pred = m_search.predict(select_x_test)

        score = m_search.score(select_x_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)

        print("Threshold = %.3f, n = %d, R2 : %.2f%%, MAE : %.2f%%" %(thresh, select_x_train.shape[1], score*100.0, mae))

        select_x_pred = selection.transform(x_pred)

        submit = m_search.predict(select_x_pred)

        a = np.arange(10000, 20000)
        submit = pd.DataFrame(submit, a)
        submit.to_csv("./dacon/comp1/submit_XGB%i_%.4f.csv"%(i, mae), header = ["hhb", "hbo2", "ca", "na"], index = True, index_label = "id")