#1. 데이터 
import numpy as np
x1 = np.array([range(1,101), range(311,411)])       # 100x2
x2 = np.array([range(711,811), range(711,811)])     # 100X2

y1 = np.array([range(101,201), range(411,511)])
y2 = np.array([range(501,601), range(711,811)])
y3 = np.array([range(411,511), range(611,711)])

#################################################
##### 여기서 부터 수정하여 소스를 완성하세요. ######
#################################################

x1 = np.transpose(x1) 
x2 = np.transpose(x2) 
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split    
x1_train, x1_test, y1_train, y1_test = train_test_split(    
    # x, y, random_state=66, shuffle=True,
    x1, y1, shuffle=False,
    train_size=0.8 
)   

from sklearn.model_selection import train_test_split    
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False,
    train_size=0.8
)

# train_test_split 매개변수 하나라도 상관 없음. x, y 짝지을 필요 없음.
from sklearn.model_selection import train_test_split    
y3_train, y3_test = train_test_split(
    y3, shuffle=False,
    train_size=0.8
)

#2. 모델구성                      
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

# 모델 2개 만들기
## input 2, output 3

input1 = Input(shape=(2, ))   
dense1_1 = Dense(5, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(3, activation = 'relu', name = 'dense1_2')(dense1_1) 

input2 = Input(shape=(2, ))   
dense2_1 = Dense(8, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(4, activation = 'relu', name = 'dense2_2')(dense2_1) 

from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_2, dense2_2]) 

middle1 = Dense(12, name = 'middle1')(merge1)
middle1 = Dense(16, name = 'middle2')(middle1)
middle1 = Dense(14, name = 'middle3')(middle1)

######### output 모델 구성 #########

output1 = Dense(20, name = 'output1')(middle1)
output1_2 = Dense(10, name = 'output1_2')(output1)
output1_3 = Dense(2, name = 'output1_3')(output1_2)

output2 = Dense(20, name = 'output2')(middle1)
output2_2 = Dense(10, name = 'output2_2')(output2)
output2_3 = Dense(2, name = 'output2_3')(output2_2)

output3 = Dense(40, name = 'output3')(middle1)
output3_2 = Dense(20, name = 'output3_2')(output3)
output3_3 = Dense(2, name = 'output3_3')(output3_2)

model = Model(inputs = [input1, input2], outputs=[output1_3, output2_3, output3_3])

model.summary() # 모델만 보여주는 거라 data 오류상관없이 출력은 제대로 된다. data는 반영되지 않기 때문에.


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], 
          [y1_train, y2_train, y3_train], epochs=100, batch_size=1,
          # validation_data=(x_val, y_val)
          validation_split = 0.25, verbose=1) 
       
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size = 1)
# 7개 출력 (전체, output1,2,3의 loss, output1,2,3의 metrics)

print("loss : ", loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
# print(y1_predict)
# print("============================")
# print(y2_predict)
# print("============================")
# print(y3_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):        
    return np.sqrt(mean_squared_error(y_test, y_predict))        

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1 + RMSE2 + RMSE3)/3)

# R2 구하기
from sklearn.metrics import r2_score
r2_1= r2_score(y1_test, y1_predict)  
r2_2= r2_score(y2_test, y2_predict) 
r2_3= r2_score(y3_test, y3_predict) 

print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 : ", (r2_1 + r2_2 + r2_3)/3)




"""
input1 = Input(shape=(2, ))   
dense1_1 = Dense(5, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(3, activation = 'relu', name = 'dense1_2')(dense1_1) 

input2 = Input(shape=(2, ))   
dense2_1 = Dense(8, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(4, activation = 'relu', name = 'dense2_2')(dense2_1) 

from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_2, dense2_2]) 

middle1 = Dense(12, name = 'middle1')(merge1)
middle1 = Dense(16, name = 'middle2')(middle1)
middle1 = Dense(14, name = 'middle3')(middle1)


output1 = Dense(20, name = 'output1')(middle1)
output1_2 = Dense(10, name = 'output1_2')(output1)
output1_3 = Dense(2, name = 'output1_3')(output1_2)

output2 = Dense(20, name = 'output2')(middle1)
output2_2 = Dense(10, name = 'output2_2')(output2)
output2_3 = Dense(2, name = 'output2_3')(output2_2)

output3 = Dense(40, name = 'output3')(middle1)
output3_2 = Dense(20, name = 'output3_2')(output3)
output3_3 = Dense(2, name = 'output3_3')(output3_2)

100, 1

RMSE1 :  0.0003455534627201514
RMSE2 :  0.0008285396005919507
RMSE3 :  0.0006903147675998066
RMSE :  0.0006214692769706361
R2_1 :  0.9999999964088062
R2_2 :  0.999999979354049
R2_3 :  0.999999985668136
R2 :  0.9999999871436637
"""