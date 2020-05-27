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
from keras.models import Sequential, Model # Model : 함수형 가져오겠다.
from keras.layers import Dense, Input
# model = Sequential()    
# model.add(Dense(5, input_dim = 3))                                            
# model.add(Dense(4))
# model.add(Dense(1))

input1 = Input(shape=(3, ))   # 100행 3열이니까 행 무시, 열 우선 # input layer가 구성되었다
                              # 함수형은 레이어별로 이름을 지어줘야 한다.
                              
dense1 = Dense(5, activation = 'relu')(input1) # 다음 레이어구나, 알 수 있음 // () 안에 윗단 레이어를 넣고 꽁지에 붙여주면 된다. // 얘 역시도 이름 만들어준다.
dense2 = Dense(10, activation = 'relu')(dense1) # activation 활성화 함수
dense3 = Dense(70, activation = 'relu')(dense2) #
dense4 = Dense(50, activation = 'relu')(dense3) 
dense5 = Dense(20, activation = 'relu')(dense4) 
output1 = Dense(1)(dense5)

model = Model(inputs = input1, outputs=output1)

"""
dense1로 모두 바꿔줘도 실행됨. dense 말고 x를 넣든 뭘 넣든 모두 실행 된다.

input1 = Input(shape=(3, ))  
                            
dense1 = Dense(5, activation = 'relu')(input1) 
dense1 = Dense(10, activation = 'relu')(dense1)
dense1 = Dense(70, activation = 'relu')(dense1) 
dense1 = Dense(50, activation = 'relu')(dense1) 
dense1 = Dense(20, activation = 'relu')(dense1) 
output1 = Dense(1)(dense1)

model = Model(inputs = input1, outputs=output1) # input은 input1, output은 outut1임을 알려줌 함수형 Model을 model로 쓰겠다.
"""

model.summary()

# 3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.25, verbose=1) 
       


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

"""
input1 = Input(shape=(3, ))  
                             
dense1 = Dense(5, activation = 'relu')(input1) 
dense2 = Dense(10, activation = 'relu')(dense1) 
dense3 = Dense(70, activation = 'relu')(dense2) 
dense4 = Dense(50, activation = 'relu')(dense3) 
dense5 = Dense(20, activation = 'relu')(dense4) 
output1 = Dense(1)(dense5)

model = Model(inputs = input1, outputs=output1)


1000, 10
RMSE :  6.254242410864013e-05
R2 :  0.999999999882359
"""

"""
"""
