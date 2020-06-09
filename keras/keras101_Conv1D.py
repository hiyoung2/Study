import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Dropout, Flatten, MaxPooling1D

# 1.데이터 준비

a = np.array(range(1, 101)) 
size = 5 

# 함수 재활용 굿굿
def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])       
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                              
    return np.array(aaa)                                     

dataset = split_x(a, size)                      
print(dataset.shape)         

print("=================")
print(dataset)       
print(type(dataset))  
                                               

x = dataset[:90, :4]  
print(x)

y = dataset[:90, 4]
print(y)

print("x.shape : ", x.shape) 
print("y.shape : ", y.shape) 

x = x.reshape(x.shape[0], x.shape[1], 1) 

print("x.reshape : ", x.shape)

x_predict = dataset[len(dataset)-6:, :4] 
print(x_predict) 

x_predict = np.reshape(x_predict, (6, 4, 1))

# train, test 분리
from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, shuffle=True, train_size=0.8)

print(x_train.shape) 
print(x_test.shape)

# x : (90, 40, 1)

#2. 모델 구성
model = Sequential()
# model.add(LSTM(140, activation = 'relu', input_shape=(4,1)))
model.add(Conv1D(30, 2, padding = 'same', input_shape = (4, 1)))
# model.add(MaxPooling1D())
model.add(Conv1D(50, 2, padding = 'same'))
model.add(Flatten())
model.add(Dense(110))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

model.fit(x_train, y_train, epochs = 100,
          batch_size = 1, verbose = 1,
          validation_split = 0.2, shuffle = True) 
                                                  
# 4. 평가, 예측
loss, mse = model.evaluate(x_train, y_train)

y_predict = model.predict(x_predict)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)
