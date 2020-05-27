#1. 데이터 
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)])
y = np.array(range(711,811)) 

x = np.transpose(x) 
y = np.transpose(y)

print(x)
print(x.shape)

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8 
)   

print(x_train)
# print(x_val)
print(x_test)


#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(10, input_dim = 3))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.25, verbose=0) 
         
          # verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정한다. 0, 1, 2 , 3의 값을 넣을 수 있다
          # verbose = 0
          # verbose = 1
          # verbose = 2
          # verbose = 3
          # 상세한 정도 1 > 2 > 3 > 0  



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

model.add(Dense(10, input_dim = 3))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

10, 1

RMSE :  0.00021187200876110574
R2 :  0.9999999986499324
"""