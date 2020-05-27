#1. 데이터 
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])    
x_pred = np.array([11,12,13]) 

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()           

model.add(Dense(5, input_dim = 1))  
model.add(Dense(8))                
model.add(Dense(13))
model.add(Dense(21))
model.add(Dense(34))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))         


"""
model.add(Dense(5, input_dim = 1))  
model.add(Dense(8))                
model.add(Dense(13))
model.add(Dense(21))
model.add(Dense(34))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))  
200, 1

loss :  1.3770318219030742e-12
mse :  1.3770318435871176e-12
y_predict :  [[11.000003]
 [12.000003]
 [13.000002]]
"""


#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=200, batch_size=1)  

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)   # 문제 : 같은 데이터로 평가함
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred) 
print("y_predict : ", y_pred)

"""
훈련 data와 평가 data를 구분해야함 (train, validation)
같은 데이터를 쓰면 안 됨
"""
