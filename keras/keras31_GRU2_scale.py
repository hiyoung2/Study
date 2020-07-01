from numpy import array                                     
  

#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([50, 60, 70]) 

x = x.reshape(x.shape[0], x.shape[1], 1)                    
x_predict = x_predict.reshape(1,3,1)  

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               



#2. 모델 구성
from keras.models import Sequential                        
from keras.layers import Dense, GRU   

model = Sequential()
model.add(GRU(8, activation = 'relu', input_length=3 , input_dim =1))               
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'min') 
                                                            
model.fit(x, y, epochs = 100000, callbacks=[early_stopping], verbose = 1)
                          

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             



'''
model.add(GRU(13, activation = 'relu', input_length=3 , input_dim =1))               
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(18))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

patience=100
epochs = 100000
# [[80.16886]]

'''