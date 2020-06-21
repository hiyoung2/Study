import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# data read
data = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submit = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", data.shape)     # (10000, 75)    
print("test.shape : ", x_pred.shape)    # (10000, 71)         
print("submit.shape : ", submit.shape)  # (10000, 4)

# 결측치 확인 및 처리
# 각 column 별로 결측치가 얼마나 있는지 알 수 있다
print(data.isnull().sum()) 

# 선형보간법 적용(모든 결측치가 처리 되는 건 아니기 때문에 검사가 필요하다)
data = data.interpolate() 
x_pred = x_pred.interpolate()

# 결측치에 평균을 대입
data = data.fillna(data.mean())
x_pred = x_pred.fillna(x_pred.mean())

# 결측치 모두 처리 됨을 확인
# print(data.isnull().sum()) 
# print(x_pred.isnull().sum()) 


# feature_importances 함수 위해서 만듦
# ndarray 형식은 feature_names 오류 발생해서
x_data = data.iloc[:, :-4]


# npy로 변환, 저장
np.save("./data/dacon/comp1/data.npy", arr = data)
np.save("./data/dacon/comp1/x_pred.npy", arr = x_pred)

# 1. 데이터 

# npy 불러오기
data = np.load("./data/dacon/comp1/data.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/x_pred.npy", allow_pickle = True)

print("data.shape :", data.shape)     # (10000, 75)
print("x_pred.shape :", x_pred.shape) # (10000, 71)


# 전체 data를 x, y 분리(슬라이싱)
x = data[:, :-4]
y = data[:, -4:]

print("======데이터 슬라이싱=====")
print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)
print()

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

print("x_train.shape :", x_train.shape) # (8000, 71)
print("x_test.shape :", x_test.shape)   # (2000, 71)
print("y_train.shape :", y_train.shape) # (8000, 4)
print("y_test.shape :", y_test.shape)   # (2000, 4)


# 2. 모델 구성
model = DecisionTreeRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)

loss = model.score(x_test,y_test)
mae = mean_absolute_error(y_test, y_pred)

submit = model.predict(x_pred)

print("loss :", loss)
print("mae :", mae)

'''
loss : -1.0874375252227273
mae : 2.533742500000001
'''

# 최종 파일 변환
a = np.arange(10000,20000)
submit= pd.DataFrame(submit, a)
submit.to_csv("./dacon/comp1/submit_dcTreeR.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )

# feature_importances

print(model.feature_importances_)

def plot_feature_importances_x_data(model) :
    n_features = x_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_data.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_x_data(model)
plt.show()




