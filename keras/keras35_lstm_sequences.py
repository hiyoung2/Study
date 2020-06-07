# 34번 copy
# 항상 파일명에 해당 소스의  keyword가 포함 되어 있다, 힌트!

from numpy import array                                     
#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6], 
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12], 
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])                                    

x_predict = array([50, 60, 70]) 

print("x.shape : ", x.shape) # (13, 3)
print("y.shape : ", y.shape) # (13, )                              

print(x.ndim) # 2, x data 2d(2차원, dimension = 2) / (13, 3)
print(y.ndim) # 1, y data 1d(1차원, dimension = 1) / (13, ) 현재 y의 구조는 벡터

# lstm 모델은 3차원으로 들어가야 하므로 와꾸를 맞춰줘야 한다
# 2d인 x data를 3d로 만들어주려면 reshape해야한다

#               행              열      몇 개씩 자를지
#            batch_size    timesteps feature
#               13          3         1  
# x = x.reshape(x.shape[0], x.shape[1], 1)       # x.reshape로 와꾸 맞추기 성공          

# x_predict 도 항상 와꾸 체크해줘야한다
print("x_predict.shape : ", x_predict.shape) # (3, )
                                             # x_predict가 y data와 같은 1차원, vector이다
# 현재 데이터의 구조를 보면
# x           y
# 1 2 3       4
# 2 3 4       5
# 3 4 5       6
# ...
# 40 50 60    70
# x_predict   y_predict
# 50 60 70    ?
# 이런 모양이기 때문에 x_predict 또한 x처럼 reshape을 해주어야한다
# reshape을 하지 않는다면? 실행해 본 결과,
# ValueError: Error when checking input: expected input_1 to have 3 dimensions, but got array with shape (3, 1)
# 위와 같은 오류가 뜬다, 값이 잘못 되었다는 ValueError가 발생
# 앞으로 데이터 와꾸 맞추는 문제로 자주 보게 될 오류
# 오류 내용은 input_1은 3차원이 되어야 하는데 현재 (3, 1) 3행 1열, 단순한 1차원(vector)으로 되어 있다는 것
# x_predict를  reshape하지 않은 까닭이다
# 그렇다면 x data를 애초에 reshape 하지 않으면?
# ValueError : Error when checking input: expected input_1 to have 3 dimensions, but got array with shape (13, 3)
# 당연히 오류 발생
# 그러니까 위와 같은 오류들이 발생했을 때는 shape이나 헷갈린다면 ndim을 써서 구조, 차원 통상 와꾸를 항상 주시하고 맞춰줘야 한다

# 나머지 x_predict도 와꾸 맞춰주기
x_predict = x_predict.reshape(1,3,1)

#2. 모델 구성
from keras.models import Sequential , Model                       
from keras.layers import Dense, LSTM, Input   
# model = Sequential()

input1 = Input(shape = (3, 1))   # Input에서는 행 무시!       
dense1 = LSTM(10, return_sequences = True , activation = 'relu')(input1)        # 상단 레이어 output이 다음 레이어의 input
dense2 = LSTM(10, activation = 'relu')(dense1)                                  # lstm은 3차원(행, 열 피처)을 받아야함, 차원이 달라짐
dense3 = Dense(5, activation = 'relu')(dense2)                                  # return_sequences = True : 차원 유지 가능케 함
output1 = Dense(1)(dense3)                                                      # reteurn_sequences default : false

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




'''
<<<<LSTM 2개 엮은 모델
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3, 1)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 3, 10)             480
_________________________________________________________________
lstm_2 (LSTM)                (None, 10)                840
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6

=============================================================================

<<<<< LSTM 1개
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3, 1)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 10)                480
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0


리턴 시퀀스 : 차원이 이어진다(유지된다)!!!!!가장 중요

summary에서 output node의 갯수는 feature의 갯수와 같다

13, 3, 1 feature의 갯수와 input_dim 의 수는 같다
첫 번째 레이어 none, 3, 1 (첫 번째 input layer의 차원) (13,3,1)이지만 행은 무시하기 때문에 none
두 번째 레이어 

(lstm 뿐만 아니라)output node의 갯수는 다음 layer의 feature의 갯수와 같다

LSTM parameter 계산은
(input_dim + bias + output) * 4(gate4) * output_node

bias는 레이어 하나당 하나!
따라서 항상 연산 할 때 bias는 1이다
레이어 하나에 항상 존재하는 전지전능한 bias!!!
'''