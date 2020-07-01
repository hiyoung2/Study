# keeras# 0525 day11

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터 준비
a = np.array(range(1, 11))
size = 5 # time_steps = 5


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
print(type(dataset))  

x = dataset[:6, :4] 

y = dataset[:6, 4:] # == [:, 4]
print(y)

print("x.shape : ", x.shape) # (6, 4)
print("y.shape : ", y.shape) # (6, 1)

x = x.reshape(x.shape[0], x.shape[1], 1) 
# == x.reshape(6, 4, 1)
# == x = np.resahpe(x, (6, 4, 1))
print("x.reshape : ", x.shape) # (6, 4, 1)


# 2. 모델 구성
# 부를 땐 케라스의 로드 모델을 불러야 함
from keras.models import load_model
model = load_model('./model/save_keras44.h5')

model.add(Dense(4, name = 'new01'))  # 저징되어 있던 모델에 레이어를 더 추가하고 싶으면 간단하게 add 하면 됨(시퀀셜이니까) 
model.add(Dense(1, name = 'new02'))   # 대신 이름이 중복되면 안 되므로 새로운 이름을 원하는대로 지어주면 된다
                                      # 여기서 마음대로 레이어 추가하면서 다시 hyperparameter tuning을 해 줄 수 있다
model.summary()

# 저장, 불러오기로 다른 사람의 모델도 쓸 수 있다 
# 우승 모델 그대로 가져와서 쓰면 최고? no
# 그대로 쓸 수 없다, 데이터가 다르기 때문에

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])             

model.fit(x, y, epochs = 100000, callbacks = [early_stopping], batch_size = 1, verbose = 1)

# 4. 평가, 예측

loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)

