#keras40_lstm_split1.py copy

# 실습 1. train, test 분리할 것 ( 8 : 2)
# 실습 2. 마지막 6개의 행을 predict로 만들 것 
# 실습 3. validaton을 넣을 것(train의 20%)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터 준비

a = np.array(range(1, 101)) # 1, 2, 3, ..., 100 
size = 5 # time_steps = 4 : column은 총 4개

# 함수 재활용 굿굿
def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])       
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                              
    return np.array(aaa)                                     

dataset = split_x(a, size)    # a, size 인자를 넣은 함수  split_x의 반환값을  dataset에 저장 -> 우리가 사용할 통데이터!                   
print(dataset.shape)          # (96, 5)


print("=================")
print(dataset)       
print(type(dataset))  # <class 'numpy.ndarray'> 
                      # 함수에 보면 return 값이 np.array이므로!                            

x = dataset[:90, :4]  # 인덱스는 0부터 시작! x train data 90개로 잘라야하므로 0~89 -> 90개!
print(x)

y = dataset[:90, 4]
print(y)

print("x.shape : ", x.shape) # (6, 4)
print("y.shape : ", y.shape) # (6, 1)

x = x.reshape(x.shape[0], x.shape[1], 1) 
# == x.reshape(6, 4, 1)
# == x = np.resahpe(x, (6, 4, 1))
print("x.reshape : ", x.shape) # (6, 4, 1)

x_predict = dataset[len(dataset)-6:, :4] # == x_predict = dataset[90:, :4]와 같은 건데
                                         # 전체 데이터셋에서 마지막 6개 행을 빼라고 했으므로
                                         # len(dataset) = 96, 데이터셋 리스트의 갯수에서 그냥 6개를 빼주는 것과 같다
                                         # len(dataset)-6은 96-6 = 90! 
                                         # 마지막 10개의 행을 예측값으로 두라고 한다면 -10하면 끝
                                         # 데이터 수가 엄청 많을 때에는 마지막 몇 개를 제하라고 할 때 계산하기 귀찮고 실수할 수도 있으니까
                                         # 전체 갯수에서 예측에 필요한 행 갯수만 빼주는 게 훨씬 수월!
print(x_predict) 

x_predict = np.reshape(x_predict, (6, 4, 1))
# == x_predict = x_predict.reshape(6, 4, 1) 
# 아까 이렇게 쓰니까 compile 문제 없는데 x_predict 부분에 빨간 줄 생성됐음
# 구문법, 새로운 문법 파악 문제라고 보면 됨, compile만 잘 되면 그냥 ignore, 크게 신경 안 써도 된다

# 데이터 와꾸 맞춰주기는 항상 미리 해두기??
# train_test_split 하기 전에 꼭 합시다
# x, y data 준비되면 바로  x_predict  준비하기

# train, test 분리
from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, shuffle=True, train_size=0.8)

print(x_train.shape) # (72, 4, 1) / 96 중 6은 x_predict로 빼뒀기 때문에 90 중 80%로 72 맞음 
print(x_test.shape)


#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(4,1)) )
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) # 마지막 output은 무조건 Dense여야한다

model.summary()


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

model.fit(x_train, y_train, epochs = 10000,
          batch_size = 1, verbose = 1, callbacks = [early_stopping],
          validation_split = 0.2, shuffle = True) # validation data 는 train의 20%로 설정(보통 test로 하는데 이번엔 train
                                                  # 72개의 20%면 14.4인데 머신이 알아서 내리거나 올려서 데이터 손실 되진 않음

# 4. 평가, 예측
loss, mse = model.evaluate(x_train, y_train)

y_predict = model.predict(x_predict)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)
