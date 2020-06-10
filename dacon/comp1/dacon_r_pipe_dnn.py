# dacon randomized, pipe, dnn
 
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping

from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score


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



np.save("./data/dacon/comp1/data.npy", arr = data)
np.save("./data/dacon/comp1/x_pred.npy", arr = x_pred)


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
print("x_test.shape :", x_train.shape)  # (8000, 71)
print("y_train.shape :", y_train.shape) # (8000, 4)
print("y_test.shape :", y_test.shape)   # (2000, 4)



def build_model(optimizer = 'adam', drop = 0.1) :
    
    inputs = Input(shape = (71, ), name = 'inputs')
    x = Dense(70, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(0.1)(x)
    x = Dense(100, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(0.1)(x)
    x = Dense(150, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(0.1)(x)
    x = Dense(130, activation = 'relu', name = 'hidden4')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation = 'relu', name = 'hidden5')(x)
    outputs = Dense(4, activation = 'softmax', name = 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')

    return model

def create_hyperparameters() :
    batches = [125, 256, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5] 
    epochs = [50, 100, 150, 200]
    return {"models__batch_size" : batches, "models__optimizer" : optimizers, 
             "models__drop" : dropout, "models__epochs" : epochs}


model = KerasRegressor(build_fn = build_model, verbose = 1)

pipe = Pipeline([("scaler", MinMaxScaler()), ('models', model)])

hyperparameters = create_hyperparameters()

kfold = KFold(n_splits = 5, shuffle = True)
search = RandomizedSearchCV(pipe, hyperparameters, cv = kfold)


search.fit(x_train, y_train)


y_pred = search.predict(x_test)

loss = search.score(x_test,y_test)
mae = mean_absolute_error(y_test, y_pred)

print("=========================================")
print(y_pred)
print("=========================================")

print("loss :", loss)
print("mae ", mae)

submit = pipe.predict(x_pred)
print("submit :", submit)

print("=========================================")
print("최적의 매개변수 :", search.best_estimator_)
print("=========================================")
print("최적의 매개변수 :", search.best_params_)
print("=========================================")
print("loss :", loss)
print("=========================================")
print("mae :", mae)


# a = np.arange(10000,20000)
# submit= pd.DataFrame(submit, a)
# submit.to_csv("./dacon/comp1/submit_r_pipe_dnn.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
