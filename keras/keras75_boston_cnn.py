# (?, 13)# (행, 13)
# boston  유명한 예제 데이터셋, 기초예제
# 케라스에 있는 예제 다 해서 싸이킷런에서 불러옴
# PCA

# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# 케라스랑 다르게 데이터 불러옴
# '''
# data       : x값
# target     : y값
# '''
import numpy as np
from sklearn.datasets import load_boston
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.decomposition import PCA

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(dataset)
# print(pd.DataFrame(dataset))
# column 13개
# (행, 13)
print(x)
print(y)

# print("x_train[0] : ", x_train[0])
# print("y_train[0] : ", y_train[0])

# print("x_train.shape : ", x_train.shape)   # x_train.shape :  (404, 13)
# print("x_test.shape : ", x_test.shape)     # x_test.shape :  (102, 13)
# print("y_train.shape : ", y_train.shape)   # y_train.shape :  (404,)
# print("y_test.shape : ", y_test.shape)     # y_test.shape :  (102,)

# 1. 데이터 준비

# 1.2 데이터 전처리 (x)
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

pca = PCA(n_components = 12)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled) 

print(x_pca.shape)

# train, test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, train_size = 0.8
)

print(x_train.shape)
print(x_test.shape)


x_train = x_train.reshape(404, 4, 3, 1)
x_test = x_test.reshape(102, 4, 3, 1)


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (4, 3, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2, 2), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(1, activation = 'relu'))
model.summary()


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor  = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 300, batch_size = 10, validation_split = 0.2, callbacks = [early_stopping], verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 10)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)