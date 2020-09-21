#1. 데이터 
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18]) 

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()           

model.add(Dense(4444, input_dim = 1))                                            
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(1))   

# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기
# layer는 input, output을 포함 5개 이상, node는 layer당 각각 5개 이상
# batch_size = 1
# epochs = 100 이상

#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)  

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


# loss :  1.9194868564605714

# mse :  1.9194867610931396
# [[ 9.872399]
#  [10.748988]
#  [11.625577]
#  [12.50216 ]
#  [13.378754]]
# RMSE :  1.385461418440967
# R2 :  0.04024832900577169



# model.add(Dense(1000, input_dim = 1))                                            
# model.add(Dense(5))  
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(9999)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(100000)) 
# model.add(Dense(5)) 
# model.add(Dense(1))   
# 500, 1
# RMSE :  1.0689864277732086
# R2 :  0.4286340086183372



# model.add(Dense(1000, input_dim = 1))                                            
# model.add(Dense(5))  
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(9999)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(11111)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(5)) 
# model.add(Dense(100000)) 
# model.add(Dense(5)) 
# model.add(Dense(1))   
# 100, 1
# RMSE :  1.2503148337345553
# R2 :  0.2183564082716657


'''
# R2 0.5 이하로 만들기

model.add(Dense(4444, input_dim = 1))                                            
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(1))   

model.fit(x_train, y_train, epochs=100, batch_size=1)  

RMSE :  1.1401741671897474
R2 :  0.35000143423658303
'''