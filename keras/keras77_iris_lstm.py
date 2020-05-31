# keras77_iris_lstm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 1. 데이터 준비
iris = load_iris()
x = iris['data']
y = iris['target']

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
    x, y, train_size = 0.8
)

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)

# 1.4 데이터 shape 맞추기
x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

# 2. 모델 구성
model = Sequential()

model.add(LSTM(33, return_sequences = True,  input_shape = (4, 1)))
model.add(Dropout(0.2))
model.add(LSTM(33, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(33))
model.add(Dense(66, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(133, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dropout(0.2))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# es = EarlyStopping(monitor = 'loss', patience = '20', mode = 'auto')

modelpath = './model/{epoch:02d}--{val_loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2, callbacks = [checkpoint], verbose = 1)



# 4. 평가, 예측
loss, acc  = model.evaluate(x_test, y_test, batch_size = 1)

print('loss : ', loss)
print('acc : ', acc)

# y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))

# 시각화
plt.figure(figsize = (10, 6)) 

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
plt.grid()
plt.title('loss')      
plt.ylabel('loss')      
plt.xlabel('epoch')          
plt.legend(loc = 'upper right') 

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '*', c = 'green', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '*', c = 'purple', label = 'val_acc')
plt.grid() 
plt.title('acc')      
plt.ylabel('acc')      
plt.xlabel('epoch')          
plt.legend(loc = 'upper right') 
plt.show()  

'''
loss :  0.10157027646297744
acc :  0.9666666388511658
'''
'''
model.add(LSTM(33, return_sequences = True,  input_shape = (4, 1)))
model.add(Dropout(0.2))
model.add(LSTM(33, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(33))
model.add(Dense(66, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(99, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(133, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dropout(0.2))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch = 1, val = 0.2, best = 89, 0.0324, es = x
loss :  0.0983049455836067
acc :  1.0
'''