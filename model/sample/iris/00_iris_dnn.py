# keras76_iris_dnn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

# 1. 데이터 준비
iris = load_iris()
x = iris['data']
y = iris['target']

print(type(iris))

print(x)
print(y)

print('x.shape : ', x.shape) # (150, 4)
print('y.shape ; ', y.shape) # (150,)

# 1.1 데이터 전처리
y = np_utils.to_categorical(y)
print('y.shape : ', y.shape) # (150, 3)

# 1.2 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print('x_scaled : ', x.shape) # (150, 4)

# 1.3 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 77, shuffle = True
)


print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

# 2. 모델 구성
model = Sequential()

model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

model.summary()

model.save('./model/sample/iris/iris_model_save.h5') 

# 3.컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# es = EarlyStopping(monitor = 'loss', patience = 50, mode = 'auto')

checkpointpath = './model/sample/iris/iris_check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = checkpointpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 1, callbacks = [checkpoint], validation_split = 0.2, verbose = 1)

# model.save_weights('./model/sample/iris/iris_save_weight.h5')
# model.save('./model/sample/iris/iris_checkpoint_best.h5')

# 4. 평가, 예측
loss, acc  = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

# y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))

# 시각화

# plt.figure(figsize = (10, 6)) 

# plt.subplot(2, 1, 1)
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
# plt.grid()
# plt.title('loss')      
# plt.ylabel('loss')      
# plt.xlabel('epoch')          
# plt.legend(loc = 'upper right') 

# plt.subplot(2, 1, 2)
# plt.plot(hist.history['acc'], marker = '*', c = 'green', label = 'acc')
# plt.plot(hist.history['val_acc'], marker = '*', c = 'purple', label = 'val_acc')
# plt.grid() 
# plt.title('acc')      
# plt.ylabel('acc')      
# plt.xlabel('epoch')          
# plt.legend(loc = 'upper right') 
# plt.show()  

