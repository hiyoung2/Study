#keras40 copy, Dense 모델로 만들기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터
a = np.array(range(1, 11))
size = 5 # time_steps = 5
#    X    Y
# 1 2 3 4  5
# 2 3 4 5  6
# 3 4 5 6  7
# 4 5 6 7  8
# 5 6 7 8  9
# 6 7 8 9 10


# LSTM 모델을 완성하시오
def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])       
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                              
    return np.array(aaa)                                     

dataset = split_x(a, size)    # (6, 5) 예상                    
                                                  
print("=================")
print(dataset)       
print(type(dataset))  # <class 'numpy.ndarray'> 
                      # 함수에 보면 return 값이 np.array이므로!                            

x = dataset[:, :4] # == dataset[:, 0:4]  훨씬 알아먹기 쉬운 듯! : 은 첨부터 끝까지! comma는 (n, m)
                               # 행, 열 끊어줌!
                               # 모든 행을 가져오겠다, 0부터 4 : 0, 1, 2, 3  column까지 가져오겠다!
                               # numpy에서는 [] 안에 () 없이 이렇게 쓸 수 있다 / 그냥 이대로 받아들이면 될 듯 / 익숙해져야함, 엄청 나옴, 반복적으로!
                               # 그냥 당연히 이렇게 자른다고 받아들이면 된다
print(x)

y = dataset[:, -1] # == [:, 4]
print(y)

print("x.shape : ", x.shape) # (6, 4)
print("y.shape : ", y.shape) # (6, 1)

# x = x.reshape(x.shape[0], x.shape[1], 1) 
# == x.reshape(6, 4, 1)
# == x = np.resahpe(x, (6, 4, 1))
# print("x.reshape : ", x.shape) # (6, 4, 1)

# shape에 들어가는 batch_size는 총 행의 수

#2. 모델 구성
model = Sequential()
model.add(Dense(8, activation = 'relu', input_shape=(4,)) )
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

model.fit(x, y, epochs = 10000, callbacks = [early_stopping], batch_size = 1, verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)

print("loss : " , loss)
print("mse: " , mse)
print("y_predict : " , y_predict)


