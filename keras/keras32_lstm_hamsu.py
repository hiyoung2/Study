# 함수형 모델로 리뉴얼 하시오
# 너는 어떤 쓰레기 데이터를 받아도 선을 만들어라 판단을 해라 ㅋㅋ

from numpy import array                                     
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([55, 65, 75]) 

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)     # lstm에 넣기 위해 3차원으로 만들어줌                
                            
x_predict = x_predict.reshape(1,3,1)

#2. 모델 구성
from keras.models import Sequential , Model                       
from keras.layers import Dense, LSTM, Input   
# model = Sequential()

input1 = Input(shape = (3, 1))        # input, 여기는 행 무시하고 적어준다     
dense1 = LSTM(100, activation = 'relu')(input1) 
dense2 = Dense(200, activation = 'relu')(dense1) 
dense3 = Dense(300, activation = 'relu')(dense2) 
dense4 = Dense(200, activation = 'relu')(dense3) 
dense5 = Dense(100, activation = 'relu')(dense4) 
output1 = Dense(1)(dense5)

model = Model(inputs = [input1], outputs=[output1])

model.summary()

# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

model.fit(x, y, epochs = 10000,  batch_size = 1, verbose = 1)
                          

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             

