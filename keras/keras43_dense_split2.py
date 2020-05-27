# 42번을  copy, Dense로 리뉴얼!


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터 준비
a = np.array(range(1, 101))
size = 5 # time_steps = 4

def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])       
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                              
    return np.array(aaa)                                     

dataset = split_x(a, size)    # (96, 5) 예상                    
print(dataset.shape)

print("=================")
print(dataset)       
print(type(dataset))  

x = dataset[:90, :4] 
print(x)

y = dataset[:90, 4:] # == [:, 4]
print(y)

print("x.shape : ", x.shape) # (6, 4)
print("y.shape : ", y.shape) # (6, 1)


x_predict = dataset[90:, :4]
print(x_predict)

# train, test 분리
from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, shuffle=True, train_size=0.8)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape=(4,)) )
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

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

model.fit(x_train, y_train, 
          epochs = 10000, callbacks = [early_stopping], batch_size = 1, verbose = 1,
          validation_split = 0.2, shuffle = True)


# 4. 평가, 예측
loss, mse = model.evaluate(x_train, y_train)

y_predict = model.predict(x_predict)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)
