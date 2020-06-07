#1. 데이터 
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18]) 
x_val = np.array([101, 102 ,103 ,104 ,105]) # x validation에 101 102 ... 105가 들어가 있다, 로만 이해
y_val = np.array([101, 102 ,103 ,104 ,105])

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50000, batch_size=10,
         validation_data=(x_val, y_val))  # validation_data로 x_val, y_val을 쓰겠다

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print(r"mse : ", mse)

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
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))
epochs = 30, batch_size = 1

RMSE :  0.026994268556948916
R2 :  0.9996356547325377

##2차

RMSE :  0.013158141729542303
R2 :  0.9999134316531126

##3차 (epochs = 50) 

RMSE :  0.012669472534348495
R2 :  0.9999197422328507

##4차 (epochs = 60)

RMSE :  0.007093996464846788
R2 :  0.9999748376070784

##5차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(11))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))
epochs = 60

RMSE :  0.0011746151467561048
R2 :  0.9999993101396285

##6차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(250))
model.add(Dense(150))
model.add(Dense(1))
epochs = 100, batch_size = 1

RMSE :  4.222096180186969e-06
R2 :  0.9999999999910869

##7차
model.add(Dense(25, input_dim = 1))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(1))
epochs = 150, batch_size = 1

RMSE :  2.132480599880018e-06
R2 :  0.9999999999977263

##8차
model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
epochs = 150, batch_size = 1

RMSE :  1.3486991523486091e-06
R2 :  0.9999999999990905


##9차
model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
epochs = 200

RMSE :  2.256805337180945e-06
R2 :  0.9999999999974534

##10차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
epochs = 200 batch_size = 1


RMSE :  2.3360154559928683e-06
R2 :  0.9999999999972715

##11차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
epochs = 200, batch_size = 1

RMSE :  9.314815862331223e-06
R2 :  0.9999999999566171

12차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
epchs = 50000, batch_size = 10


RMSE :  2.3937035366079583e-05
R2 :  0.9999999997135092
"""
