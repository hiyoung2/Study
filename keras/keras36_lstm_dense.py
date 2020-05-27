# 35 copy
# 앞선 35 lstm sequences 파일을 다시 DENSE MODEL로 변경
# 와꾸를 맞춰주는 작업이 필요하다

from numpy import array        
from keras.models import Sequential , Model                       
from keras.layers import Dense, LSTM, Input   

#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([55, 65, 75]) 

print("x.shape : ", x.shape) # (13, 3), 2d
print("y.shape : ", y.shape) # (13,), 1d                        
print("x_predict : ", x_predict.shape) # (3,) 1d

x_predict = x_predict.reshape(1,3) # 2차원 dENSE모델에 맞게끔 와꾸 수정
                                   # reshpe 하기 전에는 그냥 벡터, 1차원!!

'''
현재 DENSE 모델(2차원)을 구성하려면 모델에 투입 될 입력값 x는 2차원이므로 문제가 없다
y data는 그냥 결괏값이므로 모델에 투입 되는 것이 아니기 때문에 와꾸를 수정할 필요가 없다
다만, x_predict는 x의 차원에 맞게 와꾸를 맞춰주야 한다
why?
x_predict 는 예측용 데이터, x data로 훈련된 모델에 넣어줘야 하니까 당연히 모델에 맞게끔(이 소스에서는 DENSE)
와꾸를 수정해줘야한다
결론적으로 모델에 따라서 x data, x_predict data는 항상 와꾸를 맞춰줘야한다
'''
'''
Dense Model 2차원
###############################
### lstm 3차원 input : 2차원###
###dense 2차원 input : 1차원###
###############################
x.shape(13, 3)
x_predict(3, )
와꾸를 맞춰줘야 함!
x-predict를 x data 와꾸에 맞춰주기 위해서,,,

x_predict = array([50, 60, 70]) 을 x_predict = array([[50, 60, 70]]) 으로 차원을 맞춰주거나(만약 데이터가 많다면 단순 []입력은 어려움)
or
x_predict  = x.predict.reshape(1, 3) (1행 3열) reshape로 직접 맞춰줘도 된다

shape 함수로 와꾸를 잘 맞춰보자,,,
'''

#2. 모델 구성


input1 = Input(shape = (3, ))   # 행 무시     
dense1 = Dense(10, activation = 'relu')(input1)                             
dense2 = Dense(20, activation = 'relu')(dense1)                               
dense3 = Dense(30, activation = 'relu')(dense2)                                  
dense4 = Dense(20, activation = 'relu')(dense3)                                  
dense5 = Dense(10, activation = 'relu')(dense4)                                 
output1 = Dense(1)(dense5)                                                      

model = Model(inputs = [input1], outputs=[output1])

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto') 
                                                            
model.fit(x, y, epochs = 1000,  batch_size = 1, verbose = 1)
                          

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)  


