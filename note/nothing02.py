#0521

from numpy import array                                     
from keras.models import Sequential                        
from keras.layers import Dense, LSTM      

# 실습 : 아래 데이터로 모델을 완성하고 최적화 하시오!!!

#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([50, 60, 70]) 

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)                    

#2. 모델 구성
model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape=(3,1)) )
model.add(LSTM(10, activation = 'relu', input_length=3 , input_dim =1))               
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')             
                                                            
model.fit(x, y, epochs = 9000)
                            
x_predict = x_predict.reshape(1,3,1)                            

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             

