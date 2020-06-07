# 케라스로 만들기


import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

wine = np.load('./data/wine.npy', allow_pickle = True)

# 1. 데이터 준비
x_data = wine[:, 0:11]
y_data = wine[:, 11]

print(y_data)
print('x_data.shape : ', x_data.shape) # (4898, 11)
print('y_data.shape : ', y_data.shape) # (4898,)

# one-hot
y_data = to_categorical(y_data)
print('y_data.shape : ', y_data.shape) # (4898, 10)
print(y_data)

# scaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_data)
x = scaler.transform(x_data)

# PCA
pca = PCA(n_components=10)
pca.fit(x_data)
x_data = pca.transform(x_data)

print('x_data.shape : ', x_data.shape) # (4898, 10)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, shuffle = True
)

print('x_train.shape : ', x_train.shape) # (3918, 10)
print('x_test.shape : ', x_test.shape) # (980, 10)

# for LSTM
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1 )
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1 )

'''
# for Conv2D
x_train = x_train.reshape(x_train.shape[0], 5, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 5, 2, 1)
'''

# 2. 모델 구성

# Dense 모델

input1 = Input(shape = (10, ))

dense1 = Dense(30, activation = 'relu')(input1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dense(90, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(20, activation = 'relu')(dense1)

output1 = Dense(10, activation = 'softmax')(input1)

model = Model(inputs = input1, outputs = output1)

'''
# LSTM 모델
input1 = Input(shape = (10, 1))

dense1 = LSTM(100, activation = 'relu')(input1) 
dense1 = Dense(200, activation = 'relu')(dense1) 
dense1 = Dense(300, activation = 'relu')(dense1) 
dense1 = Dense(200, activation = 'relu')(dense1) 
dense1 = Dense(100, activation = 'softmax')(dense1) 

output1 = Dense(10)(dense1)

model = Model(inputs = [input1], outputs=[output1])
'''

'''
# Conv2D 모델

input1 = Input(shape = (5, 2, 1))
dense1 = Conv2D(77, (2, 2))(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(99, (3, 3) , padding = 'same')(dense1)   
dense1 = Dropout(0.2)(dense1)          
dense1 = Flatten()(dense1)

output1 = Dense(10, activation = 'softmax')(dense1)

model = Model(inputs = input1, outputs = output1)
 
model.summary()
'''

# 3. 컴파일, 훈련
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

# modelpath = 
# ck = ModelCheckpoint(filepath = './')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 1)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)

y_pred = model.predict(x_test)

print("loss : ", loss)
print("acc : ", acc)
# print(np.argmax(y_pred, axis = 1))



#############DNN
# dense1 = Dense(30, activation = 'relu')(input1)
# dense1 = Dense(50, activation = 'relu')(dense1)
# dense1 = Dense(90, activation = 'relu')(dense1)
# dense1 = Dense(100, activation = 'relu')(dense1)
# dense1 = Dense(20, activation = 'relu')(dense1)
# output1 = Dense(10, activation = 'softmax')(input1)
# epo = 100, batch = 32
# loss :  1.1717740311914562
# acc :  0.5295918583869934

#############LSTM
# dense1 = LSTM(100, activation = 'relu')(input1) 
# dense1 = Dense(200, activation = 'relu')(dense1) 
# dense1 = Dense(300, activation = 'relu')(dense1) 
# dense1 = Dense(200, activation = 'relu')(dense1) 
# dense1 = Dense(100, activation = 'softmax')(dense1) 

# output1 = Dense(10)(dense1)
# epo = 100, batch = 32
# loss :  12.53264149257115
# acc :  0.44795918464660645


#############Conv2D
# input1 = Input(shape = (5, 2, 1))
# dense1 = Conv2D(77, (2, 2))(input1)
# dense1 = Dropout(0.2)(dense1)
# dense1 = Conv2D(99, (3, 3) , padding = 'same')(dense1)   
# dense1 = Dropout(0.2)(dense1)          
# dense1 = Flatten()(dense1)

# output1 = Dense(10, activation = 'softmax')(dense1)

# model = Model(inputs = input1, outputs = output1)

# epo 100, batch 32
# loss :  1.094367596081325
# acc :  0.518367350101471
