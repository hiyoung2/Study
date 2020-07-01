#0521

from numpy import array                                     
                                                          
from keras.models import Sequential                        
from keras.layers import Dense, GRU, Dropout                   
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6]])     
y = array([4, 5, 6, 7])                                    

print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)   
x_predict = array([5, 6, 7])                                
x_predict = x_predict.reshape(1,3,1)                    

print(x.shape)                                            

#2. 모델 구성
model = Sequential()
model.add(GRU(8, activation = 'relu', input_length=3 , input_dim =1))               
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'min')      

# model.fit(x, y, epochs = 100000, callbacks = [early_stopping], batch_size = 1, verbose = 1)
model.fit(x, y, epochs = 500, batch_size = 1, verbose = 1)

                         
#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             

