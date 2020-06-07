from numpy import array                                     
#1. 데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50 ,60], 
            [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
            [90, 100, 110], [100, 110, 120], 
            [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x1_predict = array([55, 65, 75]) 
x2_predict = array([65, 75, 85])

print("x1.shape : ", x1.shape) # (13, 3)
print("x2.shape : ", x2.shape) # (13, 3)
print("y.shape : ", y.shape)   #(13, )                            


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)                    
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)                    

print(x1.shape) # (13, 3, 1)
print(x2.shape) # (13, 3, 1)


x1_predict = x1_predict.reshape(1,3,1)  
x2_predict = x2_predict.reshape(1,3,1)

# 2. 모델 구성
# input 2, outpu 1

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
# model = Sequential()

input1 = Input(shape = (3, 1))
dense1_1 = LSTM(10, activation = 'relu', name = 'dense1_1')(input1)
dense1_2 = Dense(15, activation = 'relu', name = 'dense1_2')(dense1_1)  
dense1_3 = Dense(25, activation = 'relu', name = 'dense1_3')(dense1_2)  
dense1_4 = Dense(20, activation = 'relu', name = 'dense1_4')(dense1_3)  

           
input2 = Input(shape = (3, 1))
dense2_1 = LSTM(10, activation = 'relu', name = 'dense2_1')(input2)
dense2_2 = Dense(15, activation = 'relu', name = 'dense2_2')(dense2_1) 
dense2_3 = Dense(23, activation = 'relu', name = 'dense2_3')(dense2_2)  
dense2_4 = Dense(18, activation = 'relu', name = 'dense2_4')(dense2_3)  


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_4, dense2_4])
middle1 = Dense(21)(merge1)
middle1 = Dense(24)(middle1)
middle1 = Dense(18)(middle1)

output1 = Dense(13, name = 'output1')(middle1)
output1_2 = Dense(10, name = 'output1_2')(output1)
output1_3 = Dense(1, name = 'output1_3')(output1_2)

model = Model(inputs = [input1, input2], outputs=[output1_3])  

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'min') 
                                                            
model.fit([x1, x2], y, epochs = 100000, callbacks = [early_stopping], verbose = 1)
                          

#4 4. 예측
print(x1_predict)
print(x2_predict)

y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)                                             

