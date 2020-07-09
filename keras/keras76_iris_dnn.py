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

print(type(iris)) # <class 'sklearn.utils.Bunch'>

# x = iris.data, y = iris.target으로 쓰니까 iris 부분에 빨간 줄이 생기면서 무언가를 경고
# 실행은 되니까 문제는 없는 건데,,,
# 해당 부분에 alt+F8 키로 뭐가 문제인지 보니까 Instance of 'tuple' has no 'target' member 이란 메세지가 떴다
# 구글링 해서 찾아본 결과 , x = iris['data], y = ['target'] 이라고 바꿔주니까 빨간 줄이 사라졌다!

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


print(type(x))


# 1.3 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 77, shuffle = True
)

# 1.3 데이터 shape 맞추기
# DNN 모델이라 따로 맞춰줄 필요 없음

print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape) # (120, 3)
print(y_test.shape)  # (30, 3)



# 2. 모델 구성
# 2.1 Sequential형

model = Sequential()

model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(99))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

model.summary()

'''
# 2.2 함수형

input1 = Input(shape = (4, ))
dense1 = Dense(55)(input1)
dense1 = Dense(77)(dense1)
dense1 = Dense(99)(dense1)
dense1 = Dense(88)(dense1)
dense1 = Dense(66)(dense1)
dense1 = Dense(44)(dense1)
dense1 = Dense(33)(dense1)

output1 = Dense(3, activation = 'softmax')(dense1)

model = Model(inputs = input1, outputs = output1)
'''

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# es = EarlyStopping(monitor = 'loss', patience = 50, mode = 'auto')

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
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

epo 100, val 0.2, batch 1, es x
loss :  0.019614876400736374
acc :  1.0
'''
'''
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(66))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch 1
loss :  0.1591352071983503
acc :  0.89999997615814
'''

'''
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55, activation = 'relu'))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(99, activation = 'relu'))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66, activation = 'relu'))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch = 1
loss :  0.21320179102379674
acc :  0.8666666746139526

'''

'''
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(99))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

epo = 100, batch = 1
loss :  0.3099500412284508
acc :  0.9666666388511658
'''

'''
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(111))
model.add(Dropout(0.3))
model.add(Dense(133))
model.add(Dropout(0.4))
model.add(Dense(122))
model.add(Dropout(0.2))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dense(3, activation = 'softmax'))

epo 300, es 50, val 0.3, batch 1, best 127, 0.1848(loss)
loss :  0.15025983566220777
acc :  0.9333333373069763
'''
'''
3차
model.add(Dense(33, input_shape = (4, )))
model.add(Dense(55))
model.add(Dropout(0.1))
model.add(Dense(77, activation = 'relu'))
model.add(Dense(111))
model.add(Dropout(0.3))
model.add(Dense(133, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(122, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(66))
model.add(Dense(44, activation = 'relu'))
model.add(Dense(33))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

epo 150, es x, val 0.2, batch 1, best 43, 0.0223
loss :  0.02353709747027679
acc :  1.0
'''

'''
함수형
input1 = Input(shape = (4, ))
dense1 = Dense(55)(input1)
dense1 = Dense(77)(dense1)
dense1 = Dense(99)(dense1)
dense1 = Dense(88)(dense1)
dense1 = Dense(66)(dense1)
dense1 = Dense(44)(dense1)
dense1 = Dense(33)(dense1)

output1 = Dense(3, activation = 'softmax')(dense1)

epo = 100, batch = 1, es = x, best = 71, 0.0036

loss :  0.22330012133207688
acc :  0.9666666388511658
'''

'''
input1 = Input(shape = (4, ))
dense1 = Dense(55)(input1)
dense1 = Dense(77)(dense1)
dense1 = Dense(99)(dense1)
dense1 = Dense(88)(dense1)
dense1 = Dense(66)(dense1)
dense1 = Dense(44, activation = 'relu')(dense1)
dense1 = Dense(33)(dense1)

output1 = Dense(3, activation = 'softmax')(dense1)

epo = 200, batch = 32
loss :  0.38386479020118713
acc :  0.8999999761581421
'''
'''
input1 = Input(shape = (4, ))
dense1 = Dense(55)(input1)
dense1 = Dense(77)(dense1)
dense1 = Dense(99)(dense1)
dense1 = Dense(88)(dense1)
dense1 = Dense(66)(dense1)
dense1 = Dense(44)(dense1)
dense1 = Dense(33)(dense1)

output1 = Dense(3, activation = 'softmax')(dense1)

epo = 200, batch = 32
loss :  0.21769794821739197
acc :  0.9333333373069763
'''