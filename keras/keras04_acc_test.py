#1. 데이터 준비
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])    
x_pred = np.array([11,12,13])   

# pred : predict
# 훈련 시킨 데이터 말고 다른 데이터를 machine에게 주고 예측해보라고 시키기 위해
# x_pred 생성


#2. 모델구성                      
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()           

model.add(Dense(5, input_dim = 1))  
model.add(Dense(3))                
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))         

#3. 컴파일, 훈련   / computer no. machine 훈련 시킴
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)  

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)   
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_pred) # x_pred의 예측값이 y_pred로 반환, y pred : 예측값
print("y_predict : ", y_pred)


"""
model.add(Dense(5, input_dim = 1))  
model.add(Dense(3))                
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))      

100, 1

loss :  1.3699263945454732e-12
acc :  1.0
y_predict :  [[11.000001]
 [12.000002]
 [13.000004]]

 loss :  1.4054535313334782e-12
acc :  1.0
y_predict :  [[11.      ]
 [12.      ]
 [13.000001]]
"""


