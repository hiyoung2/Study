import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# dacon_comp1 데이터 불러오기

data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submit = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", data.shape)             # (10000, 75) : x_train, x_test로 만들어야 함
print("test.shape : ", x_pred.shape)               # (10000, 71) : x_pred
print("submission.shape : ", submit.shape)   # (10000, 4)  : y_pred


# 결측치 확인
print(data.isnull().sum()) 

data = data.interpolate() 

x_pred = x_pred.interpolate()
# print(train.head())
# train = train.fillna(train.mean())
# print(train.head())

# 결측치보완(이전 값 대입)
# print(train.head())
data = data.fillna(method ='bfill')
print(data.head())

x_pred = x_pred.fillna(method = 'bfill')

# 결측치보완(평균값 대입법)
# print(train.head())
# train = train.fillna(train.mean())
# print(train.head())


# feature_importances 함수 위해서 만듦
# ndarray 형식은 feature_names 오류 발생해서
x_data = data.iloc[:, :-4]


# npy 변환, 저장
np.save("./data/dacon/comp1/data.npy", arr = data)
np.save("./data/dacon/comp1/x_pred.npy", arr = x_pred)

# npy 불러오기
data = np.load("./data/dacon/comp1/data.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/x_pred.npy", allow_pickle = True)

x = data[:, :71]
y = data[:, -4:]


print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

# train_test_split
print("======trai_test_split======")
print("x_train.shape :", x_train.shape)  # (8000, 71)
print("x_test.shape :", x_test.shape)    # (2000, 71)
print("y_train.shape :", y_train.shape)  # (8000, 4)
print("y_test.shape :", y_test.shape)    # (2000, 4)


print("x_pred.shape :", x_pred.shape) # (10000, 71)


# 2. 모델 구성
model = RandomForestRegressor()
model.fit(x_train, y_train)
loss = model.score(x_test, y_test)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)

submit = model.predict(x_pred)

### for pipe ###
# pipe = Pipeline([("scaler", RobustScaler()), ('ensemble', RandomForestRegressor())])
# pipe.fit(x_train, y_train)
# loss = pipe.score(x_test,y_test)
# y_pred = pipe.predict(x_test)
# mae = mean_absolute_error(y_test, y_pred)
# submit = pipe.predict(x_pred)

print()
print("loss :", loss)
print("mae :", mae)
print()

'''
loss : -0.044404494645248455
mae : 1.801080325000001
'''


# 최종 파일 변환
a = np.arange(10000,20000)
submit= pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp1/submit_RFR.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )


# feature_importances
def plot_feature_importances_x_data(model) :
    n_features = x_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_data.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_x_data(model)
plt.show()

'''
loss : -1.0514380673543013
mae : 2.5077675000000004
'''

'''
loss : -0.041883276540048316
mae : 1.798393225000001
'''