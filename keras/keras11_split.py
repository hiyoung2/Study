#1. 데이터 
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))  # x값, y값 준비 / w = 1, bias = 100임을 알 수 있음 (y = wx + b)

x_train = x[:60]        # : 앞에 아무 것도 없음 : 처음부터 1 ~ 59
x_val = x[60:80]
x_test = x[80:]         # : 뒤에 아무 것도 없음 : 끝까지

y_train = x[:60]        
y_val = x[60:80]
y_test = x[80:]        

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
         validation_data=(x_val, y_val))  
         

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):                   
    return np.sqrt(mean_squared_error(y_test, y_predict))                                          
print("RMSE : ", RMSE(y_test, y_predict))     

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  
print("R2 : ", r2)


"""
model.add(Dense(20, input_dim = 1))                                            
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(160))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(180))
model.add(Dense(160))
model.add(Dense(140))
model.add(Dense(120))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))
epochs = 100 batch_size = 10

RMSE :  0.01371744577934823
R2 :  0.9999943408024449

##2차 epochs=100, batch_size=1

RMSE :  0.011156271256089664
R2 :  0.9999962567702755

#3차 
model.add(Dense(2, input_dim = 1))                                            
model.add(Dense(4))
model.add(Dense(8))
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
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epochs = 500, batch_size = 1

RMSE :  1.609422539499587e-05
R2 :  0.9999999999922098

#4차
model.add(Dense(2, input_dim = 1))                                            
model.add(Dense(4))
model.add(Dense(8))
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
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

epochs = 1000, batch_size = 10
RMSE :  1.7973661788009256e-05
R2 :  0.9999999999902841
"""
