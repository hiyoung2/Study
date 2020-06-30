"""
x range(1,101)
y range(101,201)
w=1, bias=100

x range(311,411)
y range(711,811)
w=1, bias=400

x range(100) 은 0~99 
y range(100) 은 0~99 
w=1, bias=0
"""
# mlp : multi layer perceptron 
# 가장 기본적인 형태의 인공신경망(Artificial Neural Networks) 구조이며
# 하나의 입력층(input layer)
# 하나 이상의 은닉층(hidden layer)
# 그리고 하나의 출력층(output layer)로 구성된다

#1. 데이터 
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)])

y = np.array([range(101,201), range(711,811), range(100)]) 

print(x)
print("x.shape : ", x.shape) # (3, 100)

x = np.transpose(x) # 행과 열을 바꿔주는 함수 : transpos, 바꾸고 다시 x에 집어넣어줌
y = np.transpose(y)

print("x_trans : ", x)

print("x_trans.shape : ", x.shape) # (100, 3)

# data preprocssing
from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    # x, y, random_state=66, shuffle=True,
    x, y, shuffle=False,
    train_size=0.8  # train 80행 3열 (행의 80%로 잘림)
)   

print(type(x_train))    # 출력 : <class 'numpy.ndarray'>

# x_val, x_test, y_val, y_test = train_test_split(    
#     # x_test, y_test, random_state=66,
#     x_test, y_test, shuffle=False,
#     test_size=0.5   # test 20행 3열
# )        

# x_train = x[:60]      
# x_val = x[60:80]
# x_test = x[80:]         

# y_train = x[:60]        
# y_val = x[60:80]
# y_test = x[80:]        

# print(x_train)
# # print(x_val)
# print(x_test)


#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    

model.add(Dense(9, input_dim = 3))                                            
model.add(Dense(27))
model.add(Dense(54))
model.add(Dense(81))
model.add(Dense(90))
model.add(Dense(81))
model.add(Dense(54))
model.add(Dense(27))
model.add(Dense(9))
model.add(Dense(3))


#3. 훈련  (validation fit에 추가)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
        #validation_data=(x_val, y_val)
         validation_split = 0.3)  # train set의 0.n(n0%)


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



'''
model.add(Dense(10, input_dim = 3))                                            
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(3))
epochs = 1000, batch_size = 100

RMSE :  6.78687426848255e-05
R2 :  0.9999999994416768


model.add(Dense(9, input_dim = 3))                                            
model.add(Dense(27))
model.add(Dense(81))
model.add(Dense(90))
model.add(Dense(81))
model.add(Dense(27))
model.add(Dense(9))
model.add(Dense(3))
epochs = 1000 batch_size = 10

RMSE :  4.9679017932433184e-05
R2 :  0.9999999997008479
'''

