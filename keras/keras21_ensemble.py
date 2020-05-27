#1. 데이터 
import numpy as np
x1 = np.array([range(1,101), range(311,411), range(100)])                       # x1 현재 shape : 3 x 100(3행 100열)
y1 = np.array([range(711,811), range(711, 811), range(100)])                    # y1 현재 shape : 3 x 100(3행 100열)    

x2 = np.array([range(101,201), range(411,511), range(100,200)])                 # x2 현재 shape : 3 x 100(3행 100열)
y2 = np.array([range(501,601), range(711, 811), range(100)])                    # y2 현재 shape : 3 x 100(3행 100열)
                                                                                # 우리가 필요한 형태로 shape 변형하기
x1 = np.transpose(x1)                                                           # 100, 3 : 100행 3열(열 : data의 종류, 열 우선! 행 무시ㅠ)
x2 = np.transpose(x2)                                                           # 마찬가지 100, 3
y1 = np.transpose(y1)                                                           # y1, y2도 마찬가지 100, 3 
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split                            # train과 test로 split
x1_train, x1_test, y1_train, y1_test = train_test_split(                        # x1 - y1 짝으로 같이 적어줬는데 알고보니...
    x1, y1, random_state = 77, shuffle=True,                                    # 모든 데이터 한 번에 같이 써 줘도 잘 돌아간다고 한다
    train_size=0.8                                                              # 소스가 길어지긴 하는데 이렇게 적어주는 게 보기에 뭔가 깔끔
)   

from sklearn.model_selection import train_test_split                            # 이 소스도 shuffle = False에서 True로 다시 설정
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state = 77, shuffle = True,
    train_size=0.8
)

#2. 모델구성                      
from keras.models import Sequential, Model                                      # Sequential로 구현할 수 없는 모델, 함수형으로 만들어주자, Model을 import
from keras.layers import Dense, Input                                           # Dense와 Input layers를 import

# 모델 2개 만들기(x1과 x2 각각 진행 - 중간 병합 - 다시 y1, y2 분리)
# input 3, output 3 -> key point
input1 = Input(shape=(3, ))                                                     # 함수형은 input_dim = 3 안 씁니다, Input(sahpe=_(3,) )!!!
dense1_1 = Dense(5, activation = 'relu', name = 'dense1_1')(input1)             # 레이어 이름을 내가 보기 쉽게 바꾸고 싶으면 name 파라미터를 dense에 입력해 준다. '' 안에.
dense1_2 = Dense(4, activation = 'relu', name = 'dense1_2')(dense1_1)           # hidden layers 에 해당하는 layer들은 마음대로 추가하거나 제거 가능
                                                                                # layer들의 각각의 input layer들이 무엇인지 말할? 볼 줄 알아야 함
input2 = Input(shape=(3, ))                                                     # input2에 해당하는 데이터도 역시 column = 3!!
dense2_1 = Dense(9, activation = 'relu', name = 'dense2_1')(input2)             # node는 9개, 활성화함수로 'relu' 사용, 이름은 dense2_1로 지정, input2에서 이어진다    
dense2_2 = Dense(6, activation = 'relu', name = 'dense2_2')(dense2_1) 

# 엮어주는 것 불러오기
from keras.layers.merge import concatenate                                      # concatenate : link (things) together in a chain or series / 사슬처럼 잇다 / 단순병합!!
merge1 = concatenate([dense1_2, dense2_2])                                      # 2개 이상은 list, 대괄호[]로 항상 묶어준다(파이썬 문법)
                                                                                # input1, input2 각각의 마지막 layers를 단순 병합 해준다(가중치들의 병합이 이뤄짐)
middle1 = Dense(10, name = 'middle1')(merge1)                                   # 병합한 후 또 hidden layer 격인 middle1이란 이름의 layer들을 생성
middle1 = Dense(8, name = 'middle2')(middle1)                                   # 있어도 그만, 없어도 그만. hyper parameter tuning에 해당
middle1 = Dense(2, name = 'middle3')(middle1)                                   # 사실 이름을 똑같이 다 설정해도 machine이 학습하는데는 아무런 문제가 없다
                                                                                # 보는 우리가 헷갈릴 수 있기 때문에 name으로 이름을 설정해주면 좋다
######### output 모델 구성 #########

output1 = Dense(20, name = 'output1')(middle1)                                  # 병합과 몇 개의 hidden layers를 거치고 ouput1과 ouput2로 분리된다
output1_2 = Dense(6, name = 'output1_2')(output1)                               # ouput에서도 역시 여러 layer들을 만들 수 있다
output1_3 = Dense(3, name = 'output1_3')(output1_2)                             # 주의사항) y1의 column = 3, 따라서 가장 마지막 ouput lyaer(여기선 ouput1_2) node 갯수 3으로!!

output2 = Dense(10, name = 'output2')(middle1)
output2_2 = Dense(7, name = 'output2_2')(output2)
output2_3 = Dense(3, name = 'output2_3')(output2_2)                             # 역시 마찬가지로 node의 갯수에 주의!

model = Model(inputs = [input1, input2], outputs=[output1_3, output2_3])        # input1, 2를 함께 적어주고 ouput1_3, output2_3도 함께 묶어 적어준다
                                                                                # input과 output에 해당하는 layer들을 밝혀주면서 machine에게 Model의 짜임새를 알려준다
model.summary()



#3. 훈련  
model.compile(loss='mse', optimizer='adam', metrics=['mse'])                     # 회귀모델은 metrics = mse로!!
model.fit([x1_train, x2_train],                                                  # 2개 이상일 땐 뭐다? 리스트 []!!
          [y1_train, y2_train], epochs=100, batch_size=1,                        # 훈련 100번, 1개씩 끊어서 (데이터 양이 엄청 많다면 batch_size는 그에 맞게 크게 잡는 것이 좋을 것 같다)
          validation_split = 0.25, verbose=1)                                    # 6:2:2 data 분리 비율, verbose 1을 디폴트로 하자 우리는.

    
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size = 1)    # 평가, 예측 단계를 주석처리하고 훈련 단계의 출력 화면을 살펴보면
                                                                                 # metrics에 loss, output1_3_loss, output2_3_loss, output1_3_mse, output2_3mse 이렇게 총 5개가                                            
                                                                                 # 이렇게 총 5개가 보여진다, 
                                                                                 # 만약 이전 소스들처럼 loss, mse = model.evaluate~~~ 를 그대로 쓴다면 error가 발생
                                                                                 # why? 값이 5개가 나오는데 loss, mse 이 2개 아래에 나뉘어 들어갈 수가 없기 때문이다
                                                                                 # 따라서 아싸리 loss 하나만 적어주고 반환 값을 통째로 넣어주거나
                                                                                 # 변수를 5개 써 줘서 하나씩 집어 넣어줘야 한다
                                                                                 # data 형태를 보고 값이 5개인지 7개인지 알 수는 있지만 귀찮으니까 그냥 하나만 적어줘도 될 듯
                                                                                 
                                                                                 # a, b, c, d, e = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size = 1) 
                                                                                 # 이렇게 해도 오류없이 출력됨

y1_predict, y2_predict = model.predict([x1_test, x2_test])                       # 예측값 출력
print(y1_predict)
print(y2_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):                                                      # y_test, y_predict 는 그냥 매개변수  
                                                                                  # y1_test를 y_test 자리에 넣는 거고, y1_predict를 y_predict 자리에 넣는 것만 의미.              
    return np.sqrt(mean_squared_error(y_test, y_predict))                         # 매개가 되어주는 변수?

RMSE1 = RMSE(y1_test, y1_predict)                                                 # RMSE 함수가 정의되었으므로, 인자들을 넣어서 각각의 RMSE를 구해주면 된다
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)                                               # y1과 y2 모두 반영한 RMSE를 알기 위해 평균값을 출력한다

# R2 구하기
from sklearn.metrics import r2_score                                              # r2는 스킷런에 준비되어 있기 때문에 따로 함수를 정의할 필요가 없다 굿굿  
r2_1= r2_score(y1_test, y1_predict)  
r2_2= r2_score(y2_test, y2_predict) 

print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2)                                                   # 마찬가지 r2도 평균값을 출력
