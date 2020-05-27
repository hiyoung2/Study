# keras14_mlp1을 Sequential 에서 함수형으로 변경
# earlyStopping 적용

#1. 데이터 
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711,811), range(100)]) 

x = np.transpose(x) 
y = np.transpose(y)

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, random_state=77, shuffle=True,
    train_size=0.8 
)   

print(x_test)
print(y_test)

print(x_train.shape)
print(y_test.shape)

#2. 모델구성                      
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))   
dense1_1 = Dense(6, activation = 'relu', name = 'dense1_1')(input1) 
dense1_2 = Dense(9, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(15, activation = 'relu', name = 'dense1_3')(dense1_2) 
dense1_4 = Dense(12, activation = 'relu', name = 'dense1_4')(dense1_3) 

output1 = Dense(17, name = 'output1')(dense1_3)
output1_2 = Dense(13, name = 'output1_2')(output1)
output1_3 = Dense(3, name = 'output1_3')(output1_2)

model = Model(inputs = input1, outputs = output1_3)

model.summary()

#3. 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto') 

model.fit(x_train, y_train, epochs=1000000, batch_size=1,
           validation_split = 0.3, verbose = 1,
           callbacks=[early_stopping]) 


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

