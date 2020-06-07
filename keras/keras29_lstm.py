# 오늘은 lstm 모델 구성만 맛보기
# train_test_split 등은 나중에 또 하나 하나씩 추가함
# 현재 데이터는 split 할 것도 없다네
# 사실상 LSTM 한 줄만 코드에 삽입 된 건데 여러 용어들 때문에 공부할 게 많음


from numpy import array                                     # == import numpy as np
                                                            #    x = np.array
from keras.models import Sequential                         # keras Sequential 모델 하겠다
from keras.layers import Dense, LSTM                        # keras Dense, Lstm 쓰겠다

#1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5 ,6]])     # 바로 x = array로 데이터 입력
y = array([4, 5, 6, 7])                                     # (4, ) : input_dim = 1  
                                                            # (4, ) != (4, 1) / 4, 는 4행 1열이 아니다
                                                            # (4, )의 비밀? 
                                                            # Scalar, Vector, Matrix, Tensor 의 개념에 대해서 알아야 한다(아래에 정리)
"""
1. Scalar, 스칼라 : 하나의 숫자, 수이다.
차원은 0차원, 이미지를 예로 들면 하나의 픽셀값이라고 할 수 있다. 
그냥 상수 하나하나들을 가리킨다고 생각하면 된다

2. Vector, 벡터 : 벡터는 여러 개의 수를 나열한 것이다. 즉, 벡터는 숫자, scalar의 배열이다
벡터의 한 성분은 그 성분의 위치를 뜻하는 인덱스와 연결된다. 
벡터의 종류
열벡터 : 행렬에서 세로 줄로 늘어서는 요소로 이루어지는 벡터 (줄벡터)
수벡터 : 몇 개의 수를 가로세로로 배열한 것, n개의 수를 가로로 배열한 것을 n차원의 행벡터, 세로로 배열한 것을 n차원의 열벡터라고 한다
행벡터 : 한 행이 n열인 행렬, 행렬에서 한 행에 배열된 수를 요소로 하는 벡터
[1, 2, 3] : 행벡터
[1
2
3] : 열벡터 
벡터는 그냥 1차원의 배열이다
[] -> 벡터!

3. Matrix, 행렬 : Vector의 배열 2차원의 배열 (2차원의 Tensor 또는 2D Tensor)
Am,1 Am,2 .... A1,n
Am,1 Am,2 .... A2,n
...
Am,1 Am,2 .... Am,n -> m행 n열, 행렬

4. Tensor, 텐서 : 3차원 이상의 배열
[[1 2] [3 4]
 [5 6] [7 8]]

"""


# y2 = array([[4, 5, 6, 7]])                                # 출력 : (1, 4)
# y3 = array([[4], [5], [6], [7]])                          # 출력 : (4, 1) != (4, )


# shape는 항상 옆에 주석으로 달아두자
print("x.shape : ", x.shape)                                # x.shape :  (4, 3) : 
print("y.shape : ", y.shape)                                # y.shape :  (4, ) : 스칼라 4개 

# (1) x = x.reshape(4, 3, 1)                                # (4, 3) 에서 (4, 3, 1)로 바뀜 -> 1은 왜 ? 하나씩 잘라서 계산 하겠다
x = x.reshape(x.shape[0], x.shape[1], 1)                    # x.shape[0] : 4, x.shpae[1] : 3 -> (4, 3)의 인덱스로 생각?
                                                            # 위의 (1) 코드가 더 간단하긴 하지만 데이터에 변화가 있으면 후자가 훨씬 편리
print(x.shape)                                              # reshape의 검산 : 4*3, 4*3*1  이 같은지만 비교하면 된다
                                                            # 연속된 작업에 대해서 한개씩 작업을 하겠다! 4행 3열짜리인데 한개씩 작업을 하겠다
                                                            # 작업할 단위를 설정하려면 reshape로 구조를 변형 해 준다 
                                                            # 지금은 데이터의 수가 적지만 데이터가 많다면 x data를 2개 혹은 4개씩 끊어서 작업 할 수 있음
                                                            # 그 때엔 x = x.reshape(x.shape[0], x.shpae[1], 2(or 4))로 reshape 해 주면 된다

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1)) )
                                                            # model을 lstm으로 엮겠다
                                                            # input_shape=(3,1) : x dat의 구조는 4행 3열이었는데 4, 3, 1로 reshape했음, 거기 있는 3, 1!
                                                            # column의 갯수(3)와 몇 개씩(1) 잘라서 할 것인지
                                                            # (lstm 열에 있는 데이터를 몇 개 씩 잘라서 하겠냐, batch_size의 개념이 아님) 
                                                            # 행(4)은 여기서도 무시, column과 몇 개씩 잘라서 작업 할 것인가만 중요!!우선!!
                                                            # (None, 3, 1)
                                                            # input_shape=(3,1) 4, 3, 1 에서 행 무시, 열 우선 (3,1)을 모델의 와꾸로 잡겠다는 의미
                                                            # 엄밀히 말하면 4가 행은 아니지만 reshape 전에 4는 행이었다
                                                            # node가 10개가 나간다 여기서부턴 Dense model이다
                                                            # 10은 LSTM의 output node의 갯수

                                                            # 예를 들어, 연봉을 data로 쓰면 일일별로 있는 연봉의 데이터를 10일씩 잘라서 연산을 한다!
                                                            # 현재 데이터에 적용하면 (4, 3, 10) 이런 식으로 쓰는 것
                                                            # 자르는 작업은? 우리가 하는 일
                                                            # 지금은 연속된 데이터 3개짜리므로 지금은 1개씩 끊음
                                                            # 범위는 내가 정하는 것!!!
                                                            # 가장 중요한 것은 와꾸 맞추기!!!!!!!!!shape!!!!!!!!!!!!!!!!
                                                            # (a, b) a :column, b :몇 개씩 잘라 연산하는지 이 전체 와꾸가 항상 맞아야 함
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(optimizer = 'adam', loss = 'mse')             # metrics 하지 않아도 상관 없음 -> 실행 시킬 때 우리가 볼 수 없음
                                                            # 소스 조금씩 변경해면서 겪어봐야하니까 이번엔 안 넣음
model.fit(x, y, epochs = 5000, batch_size = 1)

x_input = array([5, 6, 7])                                  # 새로운 데이터 5, 6, 7을 넣어서 y값을 예측해보자
print(x_input.shape)                                        # 먼저 새로운 데이터의 shape 파악하기

x_input = x_input.reshape(1,3,1)                            # 원래는 3,1 -> 1, 3, 1이 됨 / 와꾸 맞춰줘야 하니까
                                                            # 원래 data x는 3차원으로 (4, 3, 1)의 shape을 가졌었음
                                                            # input_shape(3, 1) 3 * 1 = 1 * 3 * 1 로 reshape 잘 되었는지 검산 가능
                                                            # 만약 x_input = array([[[5], [6], [7]], [[6], [7], [8]]]) 이라면
                                                            # x.input = x_input.reshape(2, 3, 1) 로 해 줘야 한다
                                                            # (2, 3, 1)은 data의 수가 6개란 말이기 때문에 현재 data와 맞지 않음
print(x_input)

"""
[[[5]
  [6]
  [7]]]
  1, 3, 1 형태가 됨 -< [[[5], [6], [7]]]과 같음.
"""
"""
              X     Y
train       1 2 3   4
            2 3 4   5
            3 4 5   6
            4 5 6   7
----------------------
prredict    5 6 7   ? -> predict된 y값 위에선 yhat이라 변수를 지정(y_predict, yyy...등등 뭘 해도 상관 없음 통상적으로 알아볼 수 있게 해야함)
"""


yhat = model.predict(x_input)
print(yhat)                                                 # yhat :  x_input에 대한 예측값 / 6 7 8 넣었을 때 어떤 값이 예측 될 지

# 결과가 좋지 못하다 : 우리가 할 수 있는 부분들 hyperparameter tuning 해줘야함
# data가 너무 적기 때문일 수도



# lstm 3차원 input : 2차원
# dense 2차원 input : 1차원

"""
과제
- parameter 480이 어떻게 나오는지?
- lstm은 어떤 방식으로 연산을 하길래?

<Scalar, Vector, Matrix, Tensor>

스칼라 :  x에선 1이 하나의 스칼라, 2도 스칼라 3도..... 6도 스칼라
벡터 : 스칼라가 이어진 것
y는 벡터가 하나. 스칼라가 4개짜리 벡터 하나.
여기서 말하는 (4, ) 4 comma는? 스칼라가 4개라는 뜻

행렬 : 2차원 텐서, 행과 열
텐서 : 차원 텐서, 3차원 이상


from numpy import array 
                        
from keras.models import Sequential 
from keras.layers import Dense, LSTM


퀴즈) 다음 데이터의 구조는?

# 1번 x = array([[1,2,3],[1,2,3]])                      (2,3)
# 2번 x = array([[[1,2],[3,4]],[[4,5],[5,6]]])          (2,2,2)
# 3번 x = array([[[1],[2],[3]],[[4],[5],[6]]])          (2,3,1)
# 4번 x = array([[[1,2,3,4]]])                          (1,1,4)
# 5번 x = array([[[[1],[2]]],[[[3],[4]]]])              (2,1,2,1)
# print(x.shape)
# 다 맞췄다 개행복

[]를 잘 봐야함
쉽게 하면 ( , , ) 오른쪽 칸에서부터 가장 작은 덩어리의 개수부터 왼쪽까지 가장 큰 덩어리 개수를 적어준다고 생각
1번은 가장 안 쪽 [] 안에 1, 2, 3 총 3개, 다음 []안에 [1, 2, 3], [1, 2, 3] 이렇게 두 덩어리, 총 2개 따라서 shape : (2, 3)
2번은 가장 안 쪽 [] 안에 1, 2 총 2개, 다음 []안에 [1, 2], [3, 4] 2개, 다음에 [[1,2],[3,4]],[[4,5],[5,6]] 두 개 따라서 shape : (2, 2, 2)
3번은 가장 안 쪽 [] 안에 1 (2, 3, 4, 5, 6) 1개, 그 다음 [] 안에 [1],[2],[3] 3개, 다음 [] 안에 [[1],[2],[3]],[[4],[5],[6]] 2개 따라서 shape : (2, 3, 1)
4번은 가장 안 쪽 [] 안에 1, 2, 3, 4 총 4개, 다음 []안에 [1, 2, 3, 4] 1개, 다음 [] 안에 [[1, 2, 3, 4]]] 1개 따라서 shape : (1, 1, 4)
5번은 가장 안 쪽 [] 안에 1(2, 3, 4, ) 총 1개, 다음 [ 안에 [1], [2] 총 2개, [[1],[2]] 총 1개, 다음 [] 안에 [[[1],[2]]],[[[3],[4]]] 2개 따라서 shape : (2, 1, 2, 1)
"""
