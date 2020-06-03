# 0525 day11
# 딥러닝 케라스의 기본 구조, 딱 4단계 기억!
# 1) 데이터 준비
# 2) 모델 구성
# 3) 컴파일, 훈련
# 4) 평가, 예측

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터
a = np.array(range(1, 11))
print(a)        # [ 1  2  3  4  5  6  7  8  9 10]
print(a.shape)  # (10,)
'''
size = 5 # time_steps = 4         # 전체 데이터를 5개씩, 그 중 time_stpes = 4(입력 데이터의 컬럼 = 4, 데이터 종류!)
#    X     Y
# 1 2 3 4  5
# 2 3 4 5  6
# 3 4 5 6  7
# 4 5 6 7  8
# 5 6 7 8  9
# 6 7 8 9 10
# 이런 형태가 되도록 split 하면 된다
# x = 6 by 4
# y = 6 by 1


# LSTM 모델을 완성하시오

# 함수 재사용, 39번 파일에서 데이터를 쪼개는데 사용한 함수
# 한 번 만들어 놓으니까 다른 데이터들 쪼개는데 얼마든지 사용이 가능하다, 굉장히 편리함

def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           # len(seq) - size + 1 = 데이터를 size 대로 나누고 나면 완성되는 구조의 총 행의 갯수
        subset = seq[i : (i+size)]                 # 데이터 100개를 size 6으로 자르면 100-6+1 = 95, 바로바로 계산되면 데이터 와꾸 파악에 도움이 된다
        aaa.append([item for item in subset])      # 비어 있는 리스트에 자른 데이터들을 하나 하나씩 추가하면서 최종 데이터 구조를 만든다    
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                               # 데이터타입을 한 번씩 봐 주자     
    return np.array(aaa)                           # 넘파이 배열 사용                              

dataset = split_x(a, size)    # (6, 5) 예상        # 데이터 쪼개는 함수의 반환값을 모델에 집어넣을 하나의 데이터 셋트 값에 넣어둔다
                                                   # dataset 역시 변수명(inyoung을 하든 beer을 하든 마음대로겠지만 나중에 누구든 식별 가능할 수 있게 통상적인 이름을 쓰자)        
                                                  
print("==============================================")
print(dataset)        # 만들어진  dataset를 확인해보자    
print(type(dataset))  # <class 'numpy.ndarray'>
                      # ndarray 클래스?
                      # Numpy의 핵심인 다차원 행렬 자료구조 클래스
                      # 실제로 파이썬이 제공하는 List 자료형과 동일한 출력 형태를 갖는다
                      # 함수에 보면 return 값이 np.array이므로!                            

x = dataset[:, :4]             # == dataset[6:, 0:4] 나는 이렇게 썼는데 저게 훨씬 직관적으로 알아먹기 쉬운 듯하다(: 은 첨부터 끝까지! comma는 (n, m))
                               # 행, 열 
                               # 모든 행을 가져오겠다, 0부터 4앞 :(항상 인덱스로 생각하자) 0, 1, 2, 3  column까지 가져오겠다!
                               # numpy에서는 [] 안에 () 없이 이렇게 쓸 수 있다 / 그냥 이대로 받아들이면 될 듯 / 익숙해져야함, 엄청 나옴, 반복적으로!
                               # 그냥 당연히 이렇게 자른다고 받아들이면 된다

y = dataset[:, 4] # [:6, 4:] 이것도 선생님이 쓴 게 훨씬 간단하고 직관적! y는 dataset 의 마지막 index 4 자리의 것만 가져오면 되니까 이렇게 작성
print(y)

print("x.shape : ", x.shape) # (6, 4) : x.shape[0] = 6, x.shape[1] = 4
print("y.shape : ", y.shape) # (6, 1) # 자나꺠나 와꾸 조심

x = x.reshape(x.shape[0], x.shape[1], 1) # REMEMBER!!!! batch_size, time_steps, feature
# == x.reshape(6, 4, 1)
# == x = np.resahpe(x, (6, 4, 1)) # 이렇게 6,4 숫자로 명시해줘도 되지만 데이터 구조가 변하거나 하면 일일이 변경해주기 번거로우니 인덱스로 편하게 해두자
print("x.reshape : ", x.shape) # (6, 4, 1)

# shape에 들어가는 batch_size는 총 행의 수

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(4,1)) ) # LSTM은 3차원, input_sahpe 3차원에 맞춰서! 앞에 행 자리는 무시된 거라고 보면 된다
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

model.fit(x, y, epochs = 100000, callbacks = [early_stopping], batch_size = 1, verbose = 1)


# 4. 평가, 예측
loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)

'''