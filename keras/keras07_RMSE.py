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

model.add(Dense(5, input_dim = 1))  
model.add(Dense(15))                
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(22))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1))        

#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)  

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred) 
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기

from sklearn.metrics import mean_squared_error # 사이킷런에서 mse를 일단 땡겨옴, 
def RMSE(y_test, y_predict):                   # 함수를 만들어줘야함 (def로 명명, RMSE는 함수의 이름(사용자정의가능)) (함수를 만드는 이유 : 재사용하기 위해)
                                               # ()안 : 입력값
    return np.sqrt(mean_squared_error(y_test, y_predict))   
                                               # 이 함수를 사용하면 반환할 값 # sqrt : 루트를 씌우겠다
print("RMSE : ", RMSE(y_test, y_predict))      # 함수명 써주고 괄호 안에 인자명만 써 주면 함수가 돌아간다

# 당연히 mse, rmse는 낮을 수록 정밀도가 높다. 

"""
model.add(Dense(5, input_dim = 1))  
model.add(Dense(15))                
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(22))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1))      

100, 1

loss :  2.3646862246096135e-12
mse :  2.3646861378734396e-12
[[11.000001]
 [12.000003]
 [13.000002]
 [14.000003]
 [15.000002]]
"""