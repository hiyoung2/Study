import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. 데이터 준비
diabetes = load_diabetes()
x = diabetes['data']
y = diabetes['target']
print(x)
print(y)

print('x.shape : ', x.shape) # (442, 10)
print('y.shape : ', y.shape) # (442,)

# 1.1 데이터 전처리 - Scaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

pca = PCA(n_components = 8)
pca.fit(x)
x_pca = pca.transform(x_scaled)
print(x_pca.shape) # (442, 9) -> (442, 8)

# 1.2 데이터 분리

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y , train_size = 0.8, random_state = 77, shuffle = True
# )

# print('x_train.shape : ', x_train.shape) # (353, 10)
# print('x_test.shape : ', x_test.shape)   # (89, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y , train_size = 0.8
)

print('x_pca_train.shape : ', x_train.shape) # (353, 9) -> 8
print('x_pca_test.shape : ', x_test.shape)   # (89, 9) -> 8

# 1.3 데이터 shape 맞추기

x_train = x_train.reshape(x_train.shape[0], 4, 2, 1 )
x_test = x_test.reshape(x_test.shape[0], 4, 2, 1)

# x_train = x_train.reshape(x_train.shape[0], 3, 3, 1 )
# x_test = x_test.reshape(x_test.shape[0], 3, 3, 1)


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(50, (2, 2), input_shape = (4, 2, 1)))
model.add(Flatten()),
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(70))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# es = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')

modelpath = './model/{epoch:02d}--{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 32, validation_split = 0.2,  verbose = 1)

# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 32)

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
plt.show()

'''
model.add(Conv2D(100, (2, 2), input_shape = (5, 2, 1)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(180))
model.add(Dense(240))
model.add(Dense(110))
model.add(Dense(70))

epo = 100, batch = 32
RSME :  56.10776284667935
R2 :  0.5011844291016216

epo = 200
model.add(Dense(150))
RSME :  56.09546859174894
R2 :  0.5014030046997611

epo = 300
RSME :  55.88196214501462
R2 :  0.5051912281486934

epo 400
RSME :  55.95647499932604
R2 :  0.5038707950989723
'''

'''
model.add(Conv2D(100, (2, 2), input_shape = (4, 2, 1)))
model.add(Flatten()),
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(180))
model.add(Dense(240))
model.add(Dense(110))
model.add(Dense(70))
model.add(Dense(1))

epo = 500, batch = 32
RSME :  53.938491469913224
R2 :  0.5050332848420032
'''
'''
model.add(Conv2D(50, (2, 2), input_shape = (4, 2, 1)))
model.add(Flatten()),
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(70))
model.add(Dense(1))

epo = 500, batch = 32
RSME :  54.824785000957604
R2 :  0.506260642746045
'''