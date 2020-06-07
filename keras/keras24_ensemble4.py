######05.19######

#keras24_ensemble.py
# 실습 / 아래 데이터로 전처리 후 모델 평가지표를 완성하시오.

#1. 데이터 
import numpy as np
x1 = np.array([range(1,101), range(301,401)])

y1 = np.array([range(711,811), range(711, 811)])
y2 = np.array([range(101,201), range(411, 511)])


print("x1.shape :", x1.shape)  # (2, 100)
print("y1.shape :", y1.shape)  # (2, 100)


x1 = np.transpose(x1) 
y1 = np.transpose(y1) 
y2 = np.transpose(y2)

print("x1_trans :", x1.shape)  # (100, 2)
print("y1_trans :", y1.shape)  # (100, 2)



#################################################
##### 여기서 부터 수정하여 소스를 완성하세요. ######
#################################################

from sklearn.model_selection import train_test_split    
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(    
    x1, y1, y2, random_state=77, shuffle=True,
    # x1, y1, y2, shuffle=False,
    train_size=0.8
)   # train_test_split, 한 번에 x1, x2, y1 데이터 넣어도 돌아감!!!!(x, y 한쌍으로 넣지 않고 몰아 넣어도 된다.)

#shape 써서 항상 확인해보자.
print(x1_train.shape) #(80, 2)
print(y1_test.shape)  #(20, 2)

# from sklearn.model_selection import train_test_split    
# x2_train, x2_test = train_test_split(
#     x2, shuffle=False,
#     train_size=0.8
# )

#2. 모델구성                      
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

input1 = Input(shape=(2, ))   # input1 = Input(shape=(2, ))아주 중요!!!
dense1_1 = Dense(6, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(9, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(11, activation = 'relu', name = 'dense1_3')(dense1_2)
dense1_4 = Dense(7, activation = 'relu', name = 'dense1_5')(dense1_3)

# from keras.layers.merge import concatenate 
# merge1 = concatenate([dense1_2, dense2_2]) 

# middle1 = Dense(21, name = 'middle1')(merge1)
# middle1 = Dense(24, name = 'middle2')(middle1)
# middle1 = Dense(18, name = 'middle3')(middle1)


output1 = Dense(4, name = 'output1')(dense1_3)
output1_2 = Dense(3, name = 'output1_2')(output1)
output1_3 = Dense(2, name = 'output1_3')(output1_2)

output2 = Dense(4, name = 'output2')(dense1_3)
output2_2 = Dense(3, name = 'output2_2')(output2)
output2_3 = Dense(2, name = 'output2_3')(output2_2)

model = Model(inputs = [input1], outputs=[output1_3, output2_3])

model.summary()

#3. 컴파일, 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x1_train, 
           [y1_train, y2_train], epochs=100, batch_size=1,
         # validation_data=(x_val, y_val)
           validation_split = 0.3, verbose=1) 

#4. 평가, 예측 

# 평가값, 피팅값 배치사이즈 동일하게 해주자. evaluate에서 배치사이즈가 안 보이는 경우도 있다(시중 책), 디폴트가 있기 때문. 그래도 우리는 표시를 해 주자.
loss = model.evaluate(x1_test, [y1_test, y2_test], batch_size = 1)
print("loss : ", loss)

y1_predict, y2_predict = model.predict(x1_test)

# print("=========================")
# print(y1_predict)
# print("=========================")
# print(y2_predict)
# print("=========================")


"""
# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))        

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1+RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)  
r2_2 = r2_score(y2_test, y2_predict)

print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1+r2_2)/2)

"""
"""
input1 = Input(shape=(2, ))   
dense1_1 = Dense(9, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(18, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(15, activation = 'relu', name = 'dense1_2')(dense1_2)

dense2_1 = Dense(9, activation = 'relu', name = 'dense2_1')(dense1_2)
dense2_2 = Dense(18, activation = 'relu', name = 'dense2_2')(dense2_1) 
dense2_3 = Dense(15, activation = 'relu', name = 'dense1_2')(dense2_2)

output1 = Dense(5, name = 'output1')(dense2_2)
output1_2 = Dense(7, name = 'output1_2')(output1)
output1_3 = Dense(2, name = 'output1_3')(output1_2)

output2 = Dense(4, name = 'output2')(dense2_2)
output2_2 = Dense(6, name = 'output2_2')(output2)
output2_3 = Dense(2, name = 'output2_3')(output2_2)

500, 1

RMSE1 :  0.05781306817445701
RMSE2 :  0.018541037564268184
RMSE :  0.0381770528693626
R2_1 :  0.9998994781698723
R2_2 :  0.9999896610504072
R2 :  0.9999445696101398
"""

