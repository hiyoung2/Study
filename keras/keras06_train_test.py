#1. 데이터 
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18]) 

#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()           

model.add(Dense(5, input_dim = 1))  
model.add(Dense(100)) 
model.add(Dense(200)) 
model.add(Dense(400)) 
model.add(Dense(800))
model.add(Dense(1000))
model.add(Dense(900))
model.add(Dense(600)) 
model.add(Dense(500)) 
model.add(Dense(300)) 
model.add(Dense(150)) 
model.add(Dense(1))         

#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1)  

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred) 
print("y_predict : ", y_pred)

"""
epochs = 200, batch_size = 1
model.add(Dense(5, input_dim = 1))  
model.add(Dense(100)) 
model.add(Dense(200)) 
model.add(Dense(400)) 
model.add(Dense(800)) 
model.add(Dense(600)) 
model.add(Dense(500)) 
model.add(Dense(300)) 
model.add(Dense(150)) 
model.add(Dense(1))         


loss :  2.7284841053187847e-12
mse :  2.7284841053187847e-12
y_predict :  [[16.00001 ]
 [17.      ]
 [18.000006]]
"""


