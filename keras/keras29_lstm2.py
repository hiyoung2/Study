#0521

from numpy import array                                     
                                                          
from keras.models import Sequential                        
from keras.layers import Dense, LSTM                       
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6]])     
y = array([4, 5, 6, 7])                                    

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)                               

x = x.reshape(x.shape[0], x.shape[1], 1)                    
print("x.shape(reshape) : ", x.shape)  

'''
x의 shape = (batch_size, timesteps, feature) : 배치사이즈, 타임스텝스, 특성
feature 가장 중요 : input_dim에 위치, 이 코딩에서 feature = 1
위의 용어들을 (행, 열, 몇 개씩 자르는지) 라고 설명하심
batch_size : 행 부분 자르는 단위
feature : (1 2 3) 안에서 어떻게 자를 것인가 feature = 1이면 1 / 2 / 3, feature = 3이면 1 2 3 / feature = 2 는 불가능? 하진 않지만(오류가 나진 않음) 왜곡 발생 가능
batch_size = 1 이면 (1 2 3) / (2 3 4) / (3 4 5) / (4 5 6) 자르고 batch_size = 2 이면 (1 2 3) (2 3 4) / (3 4 5) (4 5 6)

데이터에서 100개 있을 때, train 70, test 30 으로 split 하고
batch_size 를 2로 설정했을 때 epoch 한 번에 행 2개씩 35번 train이 이뤄진다 
####################완전 중요################################
                행          열      몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
#############################################################
input_shape = (timesteps, featue)
input_length = timesteps, input_dim = feature   두 가지는 같은 의미

input_shpae 에서는 행을 무시 , 모양만 달라, (timesteps(통상,열), feature)
input_length = timesteps, input_dim = feature

1 2 3 batch_size = 5, timesteps = 3, feature = 1
2 3 4 
3 4 5 
4 5 6

1일  2일 3일 4일 5일
100 200 150 180 190 
6일 7일 8일 9일 10일
800 850 700 750 900
====================== 삼성전자 주가 data는 이렇게 잘라져 있지 않음!

우리가 잘라줘야함
5일치씩 시간순서대로 잘라주니까 "time"steps  현재 코드에서 timesteps는 3
위의 예시에서는 timesteps는 5

lstm은 데이터양이 늘어나는 게 아니라 연산양이 늘어나는 것!
우리가 이제부터는 정말 많은 양의 데이터를 다룰텐데 그 때는 GPU를 써야함 하루종일 걸릴 수도...

정제된 데이터는 거의 없다
우리가 거진 다 해야함..

transpose를 배웠으나..
data를 정리할 때 우리가 알아서 구조를 맞춰주면 되니까
무조건 transpose 해야하는 게 아니다
강박을 갖지마라 ㅋㅋ
다음 기수에는 transpose 파일 안 할 수도...

Dense : 2차원( , )
LSTM : 3차원( , , )
CNN : 4차원( , , , )

shape과 dimension을 알 수 있는 함수

z = array([[1, 2], [2, 3], [3, 4], [4, 5]]) (z.shape = (4, 2))

print(z.shape) # (4, 2)
print(z.ndim)  # 2

z = z.reshape(z.shape[0], z.shape[1], 1)   

print(z.shape)  # (4, 2, 1) 
print(z.ndim)   #  3


****** 행을 무시하는 이유? ******
어차피 DATA 양이 명시되어 있기 때문에 (TRAIN, TEST SPLIT 할 때에도, TRAIN DATA, TEST DATA양 명시, 이미 MACHINE이 알고 있다)
따라서 중요한 건 DATA의 종류 즉, 열이다.
TIMESTEPS로 열(DATA의 종류) 또한 몇 개씩 끊을 지 정할 수 있다
그런데 만약
1 2 3 4 5 6 
6 7 8 8 9 10
이라는 x data가 있는데, timesteps를 2로 설정하면
1 2 , 3 4 , 5 6 이렇게 작업이 될 것이다
그러면 2 3 , 4 5 얘네들은 훈련이 안 이루어진다
이 부분에 관해서는 나중에 또 다룰 예정
'''

                                          

#2. 모델 구성
model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape=(3,1)) )
model.add(LSTM(3, activation = 'relu', input_length=3 , input_dim =1))               
                                                                # input_dim = 1 : 1차원 (맨 처음 코딩할 때 봤던 input_dim)
                                                                # input_length = (123) (234) (345) (456) ( ) 안의 3개를 말함
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')             

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')      

model.fit(x, y, epochs = 100000, callbacks = [early_stopping], verbose = 1)

x_predict = array([5, 6, 7])                                
x_predict = x_predict.reshape(1,3,1)                            

#4 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)                                             

