# tensorboard
# 46번 카피

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터 준비
a = np.array(range(1, 101))
size = 5 # time_steps = 5

def split_x(seq, size):                            
                                                    
    aaa = []                                      
    for i in range(len(seq) - size + 1):           
        subset = seq[i : (i+size)]                  
        aaa.append([item for item in subset])       
        # == aaa.append(subset) 더 단순하게 표현    
    print(type(aaa))                              
    return np.array(aaa)                                     

dataset = split_x(a, size)                   
                                                  
print("=================")
print(dataset)       
print(type(dataset))  

x = dataset[:, :4] 

y = dataset[:, 4] 
print(y)

print("x.shape : ", x.shape) 
print("y.shape : ", y.shape) 

x = x.reshape(x.shape[0], x.shape[1], 1) 
print("x.reshape : ", x.shape)


# 2. 모델 구성
# 부를 땐 keras의 load_model을 불러야 함
# from keras.models import load_model
# model = load_model('./model/save_keras44.h5')

model = Sequential()
model.add(LSTM(5, input_shape = (4, 1)))
model.add(Dense(3))
model.add(Dense(1, name = 'new_last'))   
                                      
# model.summary()

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, TensorBoard # TensorBoard 불러오기 완료
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)
# tb_hist :  변수명


early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')      

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])             

hist = model.fit(x, y, epochs = 100, batch_size = 1, verbose = 1, 
                 callbacks = [early_stopping, tb_hist], validation_split = 0.3) 
                                             # hist는 그냥 변수명 # fit에서  
                                             # tb_hist 를 callbacks에 입력!(실행 가능해진다) 
                                             # 다음 그래프 폴더 만들고 
                                             # cmd창에서 d: 입력
                                             # cd study 입력(d:의 하위폴더인 study로 간다)
                                             # cd graph 입력(study의 하위폴더인 graph로 간다)
                                             # tensorboard --logdir=.  입력 마치면 (띄어쓰기, 부호 주의)
                                             # 주소 하나 생성 되는데 인터넷 주소창에 치면 그래프 타란~!!
                                             # local host =   내 컴퓨터, 127.0.0.1 (비전공자들도 꼭 알고 있어라!)
                                             # 6006번 포트를 오픈하겠다!

# callbacks 의 실행결과가  graph 폴더에 생김


print(hist)                  # <keras.callbacks.callbacks.History object at 0x00000241424790C8> 가 출력됨
                             # hist 의 자료형만 출력된 것
                             # 이를 보려면? 그래프로 보면 된다!

print(hist.history.keys())   # model fit에서 반환되는 것을 hist에 저장
                             # key 딕셔너리에서 봄, 뭔가 딕셔너리틱?
                             # dict_keys(['loss', 'mse']) 가 출력됨 / loss와 metrics 설정값을 보여준다
                             # 딕셔너리는  key, value 가 짝임
                             # loss, mse에 대한 key , value 가 있다는 의미!

# 과정값을 보려면 그래프로 보는 게 좋다(시각화!!!)
import matplotlib.pyplot as plt

# plot에 x, y 또는 y 값만
# x가 아래에 y가 위에?  
# plot은 여러개 가능!
plt.plot(hist.history['loss'])           # loss를 그래프에 사용하겠다
plt.plot(hist.history['val_loss'])       # validation 안 했기 때문에 error 발생
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')      # 그래프 타이틀
plt.ylabel('loss, acc')      # y축
plt.xlabel('epoch')          # x축
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 위의 plot 값들의 이름(그래프상 나타날)을 적어준다
                                                               # 선에 대한 색깔과 설명을 보여준다
                                                               # train, val  등에 관해 검증할 수 있다
                                                               # 현재 이 모델 훈련에서는 validation 의 결과가 더 안 좋다
# plt.show()                   # matplotlib 실행! - 그래프 출력
  
# matplotlib 그래프 그려주는 것
# plt라 줄여 쓰겠다
# plot : x값과 y값을 씀
# 그래프 결과 : 하강해서 뚝 떨어짐 (epoch = 10)

loss, mse = model.evaluate(x, y)

# 4. 평가, 예측

loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)

print('loss : ', loss)
print('mse: ', mse)
print('y_predict : ', y_predict)
