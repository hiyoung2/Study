# keras16_mlp3을 Sequential 에서 함수형으로 변경
# earlyStopping 적용

#1. 데이터 
import numpy as np                                                      # numpy를 import하고 앞으로 np라고 부르겠다

x = np.array(range(1,101))                                              # x는 column 1
y = np.array([range(101,201), range(711,811), range(100)])              # y는 column 3  

x = np.transpose(x)                                                     # 행과 열을 바꿔주는 함수 : transpos, 바꾸고 다시 x에 집어넣어줌
y = np.transpose(y)                                                     # x는 (1, 100)->(100, 1) / y는 (3, 100)->(100, 3)  

from sklearn.model_selection import train_test_split                    # 스킷런.model_selection에서 train_test_split을 불러와서 쓰겠다
x_train, x_test, y_train, y_test = train_test_split(                    # x와 y 통 data를 train(훈련용)과 test(평가용)으로 split 하겠다  
    x, y, random_state=77, shuffle=True,                                # x, y 데이터들을 난수 77로 shuffle 할 것이며 train용을 데이터의 80%로 쓰겠다  
    train_size=0.8                                                      # 확실히 shuffle 하니까 rmse와 r2가 좋게 나온다  
                                                                        # shuffle 은 True가 Default, 따라서 shuffle 하고 싶지 않을 경우 False를 명시
                                                                        # random_state는 난수를 설정하는 것
)   

print(x_test)                                                           # transpose 거치고 train_test_split를 거친 x_test data 확인 
print(y_test)

# transpose 거치고 train_test_split를 거친 x_test의 행렬상태(구조, 모양) 확인
print("x_train.shape :", x_train.shape)                                                    
print("y_test.shape :", y_test.shape)

#2. 모델구성                      

from keras.models import Model                                          # 함수형 모델로 만들기 : keras에서 제공하는 models package의 Model module을 import
from keras.layers import Dense, Input                                   # keras에서 layers package의 Dense와 Input module을 불러와서 쓰겠다

# from 없이 바로 import 하는 경우들은 파이썬 자체에서 제공하는 것!
# 예를 들면 numpy (import numpy as np) 
# 참고로 as를 사용하면 이름이 긴 함수, 모듈 등을 간단하게 적을 수 있다
# train_test_split as tts 이런 식으로

input1 = Input(shape=(1, ))                                             # 현재 모델, input은 1. 
                                                                        # 함수형에서는 input_dim = 1 (X, 이렇게 쓸 수 없다!!!) 
                                                                        # >>>>>>Input(shpe=(n, ) 꼭 이렇게!<<<<<<<<<<<

                                                                        # (열만 적어준다 생각하면 쉽다, 엄밀히 말하자면 틀린 말, 추후에 이해하게 될 듯?)
                                                                        # input1은 그냥 변수로서 임의대로 이름 설정해도 되지만 통상적으로 알아먹기 쉽게 input1로 설정
dense1_1 = Dense(6, activation = 'relu', name = 'dense1_1')(input1)     # input을 지나 첫 번째 hidden layer, dense1_1
                                                                        # Dense 형태이고 노드는 6개로 설정, activation(활성화함수)는 'relu', 이름은 dense1_1로 정함
                                                                        # name = ' '은 후에 summary에서 내가 알아보기 쉽게 해당 레이어의 이름을 설정할 수 있는 파라미터!
                                                                        # dense1_1은 input1을 이어받는다
                                                                        # parameter 값은 (1+1)*6 = 12    
dense1_2 = Dense(9, activation = 'relu', name = 'dense1_2')(dense1_1) 
dense1_3 = Dense(15, activation = 'relu', name = 'dense1_3')(dense1_2) 
dense1_4 = Dense(12, activation = 'relu', name = 'dense1_4')(dense1_3) 

output1 = Dense(17, name = 'output1')(dense1_3)                         # hidden layer를 거쳐 마지막 ouput 단계
                                                                        # 바로 위 layer dense1_3이 output1의 input layer인 셈
output1_2 = Dense(13, name = 'output1_2')(output1)
output1_3 = Dense(8, name = 'output1_3')(output1_2)
output1_4 = Dense(3, name = 'output1_4')(output1_3)                     # 주의!!! 현재 y data의 column = 3, 따라서 가장 마지막 찐 ouput의 node 개수를 3으로 맞춰줘야 한다

model = Model(inputs = input1, outputs = output1_4)                     # 함수형 모델은 모델 구성을 마친 후 가장 마지막에 어떤 모델인지 명시해준다
                                                                        # 우리는 inputs은 input1, outputs은 ouput1_4인 함수형모델(Model)을 만들었고 model 이라는 이름을 사용하겠다!
                                                                        # x와 y 데이터가 각각 하나씩이므로 input과 ouput이 하나씩이다
                                                                        # inputs, outputs 이렇게 -s를 붙이는 건 아마 여러 개의 input, output이 들어가는 경우도 있기 때문인가?
                                                                        # 2020.07.01 지금 보니 그냥 문법인 것 같다

                                                                        # Sequential은 이와 반대로 가장 처음에 명시한다 / model = Sequential() 요렇게

model.summary()                                                         # summary : 정리, 요약 / 현재까지 만든 model 구조를 보여준다    

# 3. 훈련                                                               # 모델을 만들었으니 이제 훈련할 차례. training!
model.compile(loss='mse', optimizer='adam', metrics=['mse'])            # 모델을 compile한다(machine이 이해할 수 있도록 해 주는 것을 컴파일이라고 한다!)
                                                                        # loss, 손실 함수로는 mse(평균제곱법), optimizer, 최적화함수는 adam(가장 많이 씀, 결과 잘 나옴)으로 하겠다
                                                                        # metrics(구글 번역 : 측정 항목) : 훈련 과정을 mse로 보여주고 그것으로 판단하겠다             
from keras.callbacks import EarlyStopping                               # 케라스의 callbacks에서 EarlyStopping(학습 조기 종료)을 부르겠다
early_stopping = EarlyStopping(monitor='loss', patience=10,            # early_stopping이라 하겠고, loss로 판단하고 loss가 100회 흔들리면(loss의 그래프 요동) 그 때 훈련을 멈추겠다(학습 종료)
                               mode = 'auto')                           # mode는 auto, 보통 loss는 min, acc는 max로 설정하는데 헷갈리면 그냥 auto 해 주면 machine이 알아서 처리
                                                                        # 지금은 loss이므로 loss 값이 최솟값에서 요동치는 게 100번일 경우로 설정된 것 


model.fit(x_train, y_train, epochs=100, batch_size=1,                # model fit, 훈련시키겠다 / train 하는 데이터 x_train, y_train, 훈련횟수는 100000번, 1개씩 끊어서 작업
         validation_split = 0.25, verbose = 1,                          # train set 중 1/4은 validation set로 쓰겠다 (현재, train _size = 0.8이므로 validation은 20, 그러면 총 6:2:2 비율)    
         callbacks = [early_stopping])                                  # verbose = 1 정도의 훈련의 정보 알려주라
                                                                        # fit 안에 callbacks = [early_stopping] 사용. 당연하지. train(학습) 종료 시키려고 쓰는 거니까 train 과정에 써 줘야겠지.

#4. 평가, 예측                                                           # 열심히 train(학습)했으니 본 시험을 봐야 한다 -> 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)                # model을 평가하겠다(test용으로 분리된 x_test, y_test, batch_size도 위와 동일하게 적어준다)
print("loss : ", loss)                                                  # 현재 이 모델에서는 loss와 mse가 하나씩 나옴
print("mse : ", mse)                                                        

y_predict = model.predict(x_test)                                       # x_test값을 통해 y예측값 구하기
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error                          # RMSE는 MSE에 ROOT 씌운 것인데, RMSE 함수는 없어서 일단 MSE를 사이킷런에서 불러온다
def RMSE(y_test, y_predict):                                            # RMSE 함수를 정의 해 준다(def), 함수의 인자는 y_test(실젯값)과 y_predict(예측값)
    return np.sqrt(mean_squared_error(y_test, y_predict))               # RMSE함수의 반환값, np.sqrt는 루트를 씌워준다                                    
print("RMSE : ", RMSE(y_test, y_predict))     

# R2 구하기
from sklearn.metrics import r2_score                                    # R2는 다행히 사이킷런에서 제공. 스킷런 매트릭스에서 r2_score를 import
r2 = r2_score(y_test, y_predict)                                        # r2에 사용될 실젯값과 예측값 넣어주기
print("R2 : ", r2)


############# DNN : Dense Model
############# CNN, RNN, DNN 종합적으로 사용할 예정
