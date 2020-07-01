from numpy import array  
from keras.models import Sequential , Model                       
from keras.layers import Dense, LSTM, Input   


# 현재 데이터 괜찮은 상태? nope
# 한 쪽으로 치우친 데이터, 평균값 내면 이상함
# 데이터 폭이 너무 큰 이러한 데이터는 scaling을 통해 해결할 수 있다 (앞에서 train_test_split 쓴 것처럼 항상 쓰게 될 것)

#1. 데이터 준비
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000], 
            [100, 200, 300]])   # 20 30 40 을 200 300 400 으로 바꿈 - 다시 천 단위로 바꿈 # 14, 3

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400]) # 50 60 70 을 500 600 700 으로 바꿈 - 천 단위로 바꿈 

# x를 정규화(아니면표준화)를 하면서 x data를 전처리 할 때 y도 같이 해줘야 할까?
# nope

x_predict = array([55, 65, 75]) 

print("x.shape: ", x.shape)  # (14, 3)
print("y.shape: ", y.shape)  # (14, )
print("x_predcit: ", x_predict.shape)  # (3, )

x_predict = x_predict.reshape(1,3) 
print("x_predict.reshape: ", x_predict.shape) # (1, 3) # x data와 와꾸가 맞춰짐

# 표준화/정규화 할 때도 shpae 와꾸 맞춰주기
# scaler는 2차원으로 맞춰주기!!!!!!!!!!!!!!

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() 
scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
# minmax, standard를 제일 많이 씀

scaler.fit(x)           # 전처리에서 fit은 실행한다는 의미 (minmax를 실행시키겠다) : 전처리를 실행     
x = scaler.transform(x) # 실행한 값을 변환시켜라, fit하고 transform하는 순서!!!!!!!!
                        # 0과 1사이로 변형된 데이터들이 나온다

# scalr.fit에 보관 중임 변형된 데이터의 가중치들을
x_predict = scaler.transform(x_predict)

print(x)   
print(x_predict)

x = x.reshape(x.shape[0], x.shape[1], 1)   
x_predict = x_predict.reshape(1,3,1)

# 3차원 모델인 LSTM에 넣기 위해 다시 와꾸 맞춰줌 , fit 전에
print("x.shape: ", x.shape)  # (14, 3, 1)
print("x_predict.shape : ", x_predict.shape) # (1, 3, 1)



# #############minmax 실행결과
# # [4000 5000 6000]
# # [1.00000000e+00 1.00000000e+00 1.00000000e+00]
# # [1 2 3]
# # [[0.00000000e+00 0.00000000e+00 0.00000000e+00]
# # 0과 1사이로 data들이 압축이 되었다 (minmax를 통해)


# ##############standardscaler 실행결과
# [[-0.5091461  -0.52172253 -0.52813466]
#  [-0.50836632 -0.52112564 -0.52765244]
#  [-0.50758653 -0.52052876 -0.52717022]
#  [-0.50680674 -0.51993187 -0.526688  ]
#  [-0.50602695 -0.51933498 -0.52620578]
#  [-0.50524716 -0.51873809 -0.52572356]
#  [-0.50446737 -0.51814121 -0.52524134]
#  [-0.50368759 -0.51754432 -0.52475912]
#  [-0.5029078  -0.51694743 -0.5242769 ]
#  [-0.50212801 -0.51635054 -0.52379468]
#  [ 1.04965084  1.26774696  1.39930025]
#  [ 1.82943921  1.86463471  1.88152064]
#  [ 2.60922758  2.46152247  2.36374103] 2.36374103(6000) : 가장 오른쪽에 치우침
#  [-0.43194706 -0.40353876 -0.38491521]] -0.38491521 (300) : 가장 평균에 가까움


# # 우리가 배운 전처리 방법들, 지금은 하나씩 적용시켜봤지만 나중엔 다 쓰게 된다
# # train_test_split 
# # minmaxscaler, standardscaler

#2. 모델 구성

input1 = Input(shape = (3, 1))        
dense1 = LSTM(50, return_sequences = True , activation = 'relu')(input1)        # 상단 레이어 output이 다음 레이어의 input
dense2 = LSTM(100, activation = 'relu')(dense1)                                  # lstm은 3차원(행, 열 피처)을 받아야함, 차원이 달라짐
dense3 = Dense(55, activation = 'relu')(dense2)                                  # 리턴 시퀀스 : 차원 유지 가능케 함
output1 = Dense(1)(dense3)                                                      # 리턴 시퀀스 디폴트 : false

model = Model(inputs = [input1], outputs=[output1])

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')     

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode = 'auto') 
                                                            
model.fit(x, y, epochs = 100000,  callbacks = [early_stopping], batch_size = 1, verbose = 1)
                          

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             
