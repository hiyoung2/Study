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

model.add(Dense(10, input_dim = 1))  
model.add(Dense(30))
model.add(Dense(50))                
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(110))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))        

"""
model.add(Dense(10, input_dim = 1))  
model.add(Dense(50))
model.add(Dense(100))                
model.add(Dense(120))
model.add(Dense(160))
model.add(Dense(180))
model.add(Dense(150))
model.add(Dense(80))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))        
 

500, 1

loss :  3.637978807091713e-13
mse :  3.6379788613018216e-13
[[11.000002]
 [12.000002]
 [13.      ]
 [13.999999]
 [15.000001]]

"""

#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1)  

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred) 
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):                   
    return np.sqrt(mean_squared_error(y_test, y_predict))                                          
print("RMSE : ", RMSE(y_test, y_predict))     

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # 맨 앞 r2는 그냥 변수명
print("R2 : ", r2)

# R2는 1에 가까울수록 좋은 모델
# RMSE(또는 다른 것)와 같이 씀







