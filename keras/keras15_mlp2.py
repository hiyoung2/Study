# x 3, y 1 : 가장 현실적으로 많이 쓰이는 모델
# 입력값 여러 종류, 연산하는 파라미터가 3배, column이 많으면 많을 수록 낫다
# 당뇨병 확률 : 키, 몸무게, 병력, 등등의 데이터 종류 많으면 많을 수록 예측 정확도 향상

#1. 데이터 
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)])
y = np.array(range(711,811)) 

x = np.transpose(x) 
y = np.transpose(y)

print(x)

# print(x.shape)
# data preprocssing

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8 
)   
# x_val, x_test, y_val, y_test = train_test_split(    
#     # x_test, y_test, random_state=66,
#     x_test, y_test, shuffle=False,
#     test_size=0.3  
# )        

# x_train = x[:60]      
# x_val = x[60:80]
# x_test = x[80:]         

# y_train = x[:60]        
# y_val = x[60:80]
# y_test = x[80:]        

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
model.fit(x_train, y_train, epochs=100, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.25)  # train set의 0.n(n0%)


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
epochs = 100, batch_size = 1

## 1차
RMSE :  0.0003030320211744729
R2 :  0.9999999972382434

## 2차
RMSE :  0.00014635716318703366
R2 :  0.9999999993557769
"""