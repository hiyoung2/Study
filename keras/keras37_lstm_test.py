# 36 copy. DENSE로 구성된 모델보다 좋게?
# LSTM layer 5개로 모델 구성하기

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

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)                 
                            
x_predict = x_predict.reshape(1,3,1)

#2. 모델 구성


input1 = Input(shape = (3, 1))        
dense1 = LSTM(10, return_sequences = True , activation = 'relu')(input1)         # 상단 레이어 output이 다음 레이어의 input
dense2 = LSTM(20, return_sequences = True , activation = 'relu')(dense1)         # lstm은 3차원(행, 열 피처)을 받아야함, 차원이 달라짐
dense3 = LSTM(30, return_sequences = True , activation = 'relu')(dense2)         # 리턴 시퀀스 : 차원 유지 가능케 함
dense4 = LSTM(20, return_sequences = True , activation = 'relu')(dense3)         # 리턴 시퀀스 : 차원 유지 가능케 함
dense5 = LSTM(10, activation = 'relu')(dense4)                                   # 리턴 시퀀스 : 차원 유지 가능케 함
output1 = Dense(1)(dense5)                                                       # 리턴 시퀀스 디폴트 : false

model = Model(inputs = [input1], outputs=[output1])

model.summary()

# 3. 컴파일, 실행
model.compile(optimizer = 'adam', loss = 'mse')     

# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto') 
                                                            
model.fit(x, y, epochs = 1000, batch_size = 1, verbose = 1)
                          

# 4. 평가, 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             


'''
lstm 5개 엮은 것이 dense보다 왜 더 좋은 결과가 나오지 않을까
5배 더 좋아야 할텐데
연산이 지나치게 많아서?

첫 번째 lstm layer까지는 순차적으로 간다
그런데, 두 번째 lstm layer부터... 받아들이는 input 값이 순차적일까?
첫 번째에서 나온 output이 none, 3, 10 
hidden layer에서 이뤄지는 가중치값들을 우리가 알 수 있는가

(1 + 1 + 10) * 4 = 48
(10 + 1 + 10) * 3 = 84 node의 갯수
84개가 순차적 데이터?
no
y = wx + b 최적의 w값을 구하기 위한 중간의 수치일 뿐
순차적 데이터는 아니다

현재 상태에서는 두 번째 lstm layer부터는 순차적으로 output이 나오지 않는다
none, 3, 10 순차적 데이터라는 확신이 x
문제가 있을 수 있다

그래서 lstm 두 개 이상 엮었을 때 두 번째부터 순차적인지 본인이 판단을 해야 한다
but hidden layer라서 잘 알 수 없다

만들어서 돌려보고 생각,,

weight를 유추할 수 있는 정도의 쉬운 데이터는
lstm 순차적 확률이 높다

결론은 겪어봐야 안다


'''