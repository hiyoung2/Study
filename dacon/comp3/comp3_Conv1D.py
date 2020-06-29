import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout, Dense, Input
from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse


x = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
y = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)
x_pred = pd.read_csv("./data/dacon/comp3/test_features.csv", header = 0, index_col = 0)
submission = pd.read_csv("./data/dacon/comp3/sample_submission.csv", header = 0, index_col = 0)

print("x.shape :", x.shape)    # (1050000, 5)
print("y.shape :", y.shape)    # (2800, 4)
print("x_pred.shape :", x_pred.shape)    # (262500, 5)
print("submission :", submission.shape)  # (700, 4)


print(x.isnull().sum())
'''
Time    0
S1      0
S2      0
S3      0
S4      0
'''

print(x.info())
print(x.describe())
print("===========================================================================================")

# time data를 일단 슬라이싱
x = x.values[:, 1:]
y = y.values
x_pred = x_pred.values[:, 1:]


print("x.shape :", x.shape) # x_shape : (1050000, 4)
print("x_pred.shape :", x_pred.shape) # x_pred.shape : (262500, 4)
print("===========================================================================================")

scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(2800, 375, 4)
x_pred = x_pred.reshape(700, 375, 4)

x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 66, shuffle = True)

model = Sequential()
model.add(Conv1D(100, 2, activation = 'relu', input_shape = (375, 4)))     
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(150))
model.add(Dropout(0.2))
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dense(4))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 32)

y_pred = model.predict(x_pred)

print("y_pred.shape :", y_pred.shape)

print("loss :", loss)
print("mse :", mse)

submission = pd.DataFrame(y_pred, np.arange(2800, 3500))
submission.to_csv("./dacon/comp3/0629/submission_h_Conv1D{mse}.csv", header = ["X", "Y", "M", "V"], index = True, index_label = "id")