# day 10 0522

# keras34_lstrm_earlystopping.py
# 기존의 32lstm hamsu 파일 복붙, early_stopping 적용

from numpy import array                                     
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([50, 60, 70]) 

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)                 
                            
x_predict = x_predict.reshape(1,3,1)

#2. 모델 구성
from keras.models import Sequential , Model    # Sequential : 순차적 구성, Model : 함수형                   
from keras.layers import Dense, LSTM, Input   

input1 = Input(shape = (3, 1))        
dense1 = LSTM(100, activation = 'relu')(input1) 
dense2 = Dense(200, activation = 'relu')(dense1) 
dense3 = Dense(300, activation = 'relu')(dense2) 
dense4 = Dense(200, activation = 'relu')(dense3) 
dense5 = Dense(100, activation = 'relu')(dense4) 
output1 = Dense(1)(dense5)

model = Model(inputs = [input1], outputs=[output1])

model.summary()

# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto') 
                                                            
model.fit(x, y, epochs = 100000,  callbacks = [early_stopping], batch_size = 1, verbose = 1)
                          

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             

'''
early_stopping

226
453
325
435
218 - 81.13399
390 - 80.92485
552 - 80.948654
452 - 80.90706
313 - 77.96036
376 - 80.26248
280 - 80.652824
545 - 87.783554
'''