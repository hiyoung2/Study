#1. 데이터 
import numpy as np
x1 = np.array([range(1,101), range(311,411), range(411,511)])
x2 = np.array([range(711,811), range(711, 811), range(511,611)])

y1 = np.array([range(101,201), range(411, 511), range(100)])


print("x1.shape :", x1.shape)
print("x2.shape :", x2.shape)
print("y1.shape :", y1.shape)

x1 = np.transpose(x1) 
x2 = np.transpose(x2) 
y1 = np.transpose(y1)


print("x1_trans :", x1.shape)
print("x2_trans :", x2.shape)
print("y1_trans :", y1.shape)

#################################################
##### 여기서 부터 수정하여 소스를 완성하세요. ######
#################################################

from sklearn.model_selection import train_test_split    
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(    
    # x, y, random_state=66, shuffle=True,
    x1, x2, y1, shuffle=False,
    train_size=0.8
)   # train_test_split, 한 번에 x1, x2, y1 데이터 넣어도 돌아감!!!!

# from sklearn.model_selection import train_test_split    
# x2_train, x2_test = train_test_split(
#     x2, shuffle=False,
#     train_size=0.8
# )




#2. 모델구성                      
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

# 모델 2개 만들기
## input 3, output 3

input1 = Input(shape=(3, ))   
dense1_1 = Dense(6, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(9, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(15, activation = 'relu', name = 'dense1_3')(dense1_2) 
dense1_4 = Dense(12, activation = 'relu', name = 'dense1_4')(dense1_3) 

input2 = Input(shape=(3, ))   
dense2_1 = Dense(5, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(11, activation = 'relu', name = 'dense2_2')(dense2_1) 
dense2_3 = Dense(13, activation = 'relu', name = 'dense2_3')(dense2_2) 
dense2_4 = Dense(7, activation = 'relu', name = 'dense2_4')(dense2_3) 

from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_2, dense2_2]) 

middle1 = Dense(21, name = 'middle1')(merge1)
middle1 = Dense(24, name = 'middle2')(middle1)
middle1 = Dense(18, name = 'middle3')(middle1)


output1 = Dense(17, name = 'output1')(middle1)
output1_2 = Dense(13, name = 'output1_2')(output1)
output1_3 = Dense(3, name = 'output1_3')(output1_2)

model = Model(inputs = [input1, input2], outputs = output1_3)

model.summary()

# 3. 컴파일, 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit([x1_train, x2_train], 
           y1_train, epochs=300, batch_size=1,
         # validation_data=(x_val, y_val)
           validation_split = 0.25, verbose=1) 
       
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)

print("loss : ", loss)

y1_predict = model.predict([x1_test, x2_test])
print(y1_predict)



# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))        

RMSE = RMSE(y1_test, y1_predict)
print("RMSE : ", RMSE)
 

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)  

print("R2 : ", r2)





"""
input1 = Input(shape=(3, ))   
dense1_1 = Dense(9, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(18, activation = 'relu', name = 'dense1_2')(dense1_1) 

input2 = Input(shape=(3, ))   
dense2_1 = Dense(9, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(18, activation = 'relu', name = 'dense2_2')(dense2_1) 

from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_2, dense2_2]) 

middle1 = Dense(15, name = 'middle1')(merge1)
middle1 = Dense(27, name = 'middle2')(middle1)
middle1 = Dense(21, name = 'middle3')(middle1)

output1 = Dense(12, name = 'output1')(middle1)
output1_2 = Dense(6, name = 'output1_2')(output1)
output1_3 = Dense(3, name = 'output1_3')(output1_2)

500, 1

RMSE :  0.024779618952240248
R2 :  0.9999815329469047

#3

input1 = Input(shape=(3, ))   
dense1_1 = Dense(6, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(9, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(15, activation = 'relu', name = 'dense1_3')(dense1_2) 
dense1_4 = Dense(12, activation = 'relu', name = 'dense1_4')(dense1_3) 

input2 = Input(shape=(3, ))   
dense2_1 = Dense(5, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(11, activation = 'relu', name = 'dense2_2')(dense2_1) 
dense2_3 = Dense(13, activation = 'relu', name = 'dense2_3')(dense2_2) 
dense2_4 = Dense(7, activation = 'relu', name = 'dense2_4')(dense2_3) 

from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_2, dense2_2]) 

middle1 = Dense(21, name = 'middle1')(merge1)
middle1 = Dense(24, name = 'middle2')(middle1)
middle1 = Dense(18, name = 'middle3')(middle1)


output1 = Dense(17, name = 'output1')(middle1)
output1_2 = Dense(13, name = 'output1_2')(output1)
output1_3 = Dense(3, name = 'output1_3')(output1_2)

300,1

RMSE :  0.0018961434882145797
R2 :  0.9999998918688683
"""