# keras79_diabetes_dnn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 1. 데이터 준비
diabetes = load_diabetes()
x = diabetes['data']
y = diabetes['target']

print(x)
print(y)

print('x.shape : ', x.shape) # (442, 10)
print('y.shape : ', y.shape) # (442,)

# 1.1 데이터 전처리 - Scaler, pca
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# pca = PCA(n_components = 2)
# pca.fit(x)
# x_pca = pca.transform(x)

# x data 세 번째 column 것만 꺼내기?
# x = x[:, np.newaxis, 2]
# print(x)
# print('x_new.shape : ', x.shape)


# 1.2 데이터 분리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y , train_size = 0.8, random_state = 77, shuffle = True
)

print('x_train.shape : ', x_train.shape) # (353, 10)
print('x_test.shape : ', x_test.shape)   # (89, 10)

# print('x_train_new.shape : ', x_train.shape) # (353, 1)
# print('x_test_new.shape : ', x_test.shape)   # (89, 1)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x_pca, y , train_size = 0.8
# )

# print('x_pca_train.shape : ', x_train.shape) # x_pca_train.shape :  (353, 2)
# print('x_pca_test.shape : ', x_test.shape)   # x_pca_test.shape :  (89, 2)

# 2. 모델 구성
model = Sequential()

model.add(Dense(100, input_shape=(10,)))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(900))
model.add(Dense(1500))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

# modelpath = './model/{epoch:02d}--{loss:.4f}.hdf5'

# checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict(x_test)
print(y_pred)

# RMSE, R2

from sklearn.metrics import mean_squared_error
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('RSME : ', rmse(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print('R2 : ', r2)

# 시각화

plt.figure(figsize =(10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c ='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['mse'], marker = '*', c = 'green', label = 'mse')
plt.plot(hist.history['val_mse'], marker = '*', c = 'purple', label = 'val_mse')
plt.grid()
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
# plt.show()

'''
100
150
200
500
250
60
RSME :  56.39525835643953
R2 :  0.496059482462316
흠,,, 과제로 R2 0.5 이하로 만들기 할 땐 그렇게 안 나왔는데 뭐죠,,
'''

'''
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(5000))
model.add(Dense(90))
model.add(Dense(1))

epo = 100, batch = 1
RSME :  55.32014623931648
R2 :  0.5150904520775285 
'''
'''
model.add(Dense(10, input_shape=(10,)))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(1))

epo = 100, batch = 1
RSME :  56.50577893451295
R2 :  0.4940823526878463

2차
RSME :  55.969851680558484
R2 :  0.5036335623041546
'''

