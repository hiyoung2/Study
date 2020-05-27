#1. 데이터 
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))  # x값, y값 준비 / w = 1, bias = 100임을 알 수 있음 (y = wx + b)

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, random_state=77, shuffle=True,
    # x, y, shuffle=False,
    train_size=0.6
)   
# x_val, x_test, y_val, y_test = train_test_split(    
#     # x_test, y_test, random_state=66,
#     x_test, y_test, shuffle=False,
#     test_size=0.5
# )        

# x_train = x[:60]      
# x_val = x[60:80]
# x_test = x[80:]         

# y_train = x[:60]        
# y_val = x[60:80]
# y_test = x[80:]        


#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=500, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.3)  # train set의 0.n(n0%)

print(x_train)
# print(x_val)
print(x_test)


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
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(250))
model.add(Dense(150))
model.add(Dense(1))
epochs = 100, batch_size =1

RMSE :  7.169195532735295e-05
R2 :  0.9999999999614279


##2차
model.add(Dense(5, input_dim = 1))                                            
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(110))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))
epochs = 100, batch_size=1

RMSE :  2.5646570303301408e-05
R2 :  0.9999999999950638

##3차

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
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
epochs = 100, batch_size = 1

RMSE :  8.395801174766409e-05
R2 :  0.9999999999470999

## 4차
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
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
epochs = 5000, batch_size = 10

RMSE :  2.7718966864906234e-05
R2 :  0.999999999976892

##5차

model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
epochs = 10000 batch_size = 1

RMSE :  6.694766530908297e-05
R2 :  0.9999999998652033

##6차
model.add(Dense(10, input_dim = 1))                                            
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
epochs = 100000 batch_size = 1

RMSE :  7.075209423443378e-05
R2 :  0.9999999998494479
"""

"""
train_size = 0.6, test_size = 0.5 실행 결과 :
The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.

train_sie = 0.6, test_size = 0.3 실행 결과 :
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
 49 50 51 52 53 54 55 56 57 58 59 60]
[61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84
 85 86 87 88 89 90]
 train에서 0.6, test에서 0.3 나머지 0.1은 사용하지 않음

 train_size = 0.6, test_size = 0.2를 
 3번 실행 : R2 음수
 1번 실행 : R2 0.9824802...
 6번 실행 : R2 음수
"""