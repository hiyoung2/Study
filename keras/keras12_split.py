#1. 데이터 
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))  # x값, y값 준비 / w = 1, bias = 100임을 알 수 있음 (y = wx + b)

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, random_state=66, shuffle=True,
    # x, y, shuffle=False,
    train_size=0.8
)   
x_val, x_test, y_val, y_test = train_test_split(    
    x_test, y_test, random_state=66,
    # x_test, y_test, shuffle=False,
    test_size=0.4
)        

# x_train = x[:60]      
# x_val = x[60:80]
# x_test = x[80:]         

# y_train = x[:60]        
# y_val = x[60:80]
# y_test = x[80:]        

print(x_train)
print(x_val)
print(x_test)

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(10000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=100,
         validation_data=(x_val, y_val))  

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=100) 
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
epochs = 100, batch_size = 1
RMSE :  0.10889578031032307
R2 :  0.9996433596700934

epochs = 1000, batch_size = 10
RMSE :  5.9342762703338795e-05
R2 :  0.999999999329226

epochs = 10000, batch_size = 100
RMSE :  2.224331624925728e-05
R2 :  0.999999999905759
"""