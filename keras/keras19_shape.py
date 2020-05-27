######05.18######

#1. 데이터 준비
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)])
y = np.array(range(711,811)) 

x = np.transpose(x) # trnaspose : 전치 (행과 열을 바꿈)
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

# model.add(Dense(5, input_dim = 3)) 
# == model.add(Dense(5, input_shape(3, ))) 
# Sequential은 두 형식 모두 쓸 수 있다.
# 함수형은 input_shape만 쓸 수 있음.   
         
model.add(Dense(5, input_shape=(3, ))) # 행은 무시, 열만 적어준다.(엄밀히 말하면 틀린 거지만 앞으로 배울 것)

model.add(Dense(3))

model.add(Dense(1))


# 3. 컴파일, 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.25, verbose=0) 
  
# 4. 평가, 예측
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

