from numpy import array                                     
                                                          
from keras.models import Sequential                        
from keras.layers import Dense, SimpleRNN                      
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6]])     
y = array([4, 5, 6, 7])                                    

print("x.shape :", x.shape) # (4, 3)
print("y.shape : ", y.shape) # (4, )                        

x = x.reshape(x.shape[0], x.shape[1], 1)                    
print("x.reshape :", x.shape) # (4, 3, 1)

x_predict = array([5, 6, 7])
print("x_predict.shape :", x_predict.shape) # (3, )                                
x_predict = x_predict.reshape(1,3,1)     
print("x_predict.reshape :", x_predict.shape) # (1, 3, 1)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(8, activation = 'relu', input_length=3 , input_dim =1))               
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'mse')   

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'min') 

model.fit(x, y, epochs = 100000, callbacks = [early_stopping], verbose = 1)


# 4. 평가, 예측
# print(x_predict)
y_predict = model.predict(x_predict)
# print(y_predict)                                             

# [[8.015787]]