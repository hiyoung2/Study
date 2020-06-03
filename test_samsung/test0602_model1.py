# ensemble로 모델을 짜야 한다
# 일반적인 Dense Model 두 가지를 병합

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# CNN이 LSTM 보다 시계열에 더 좋은 경우도 있다
# CNN은 LSTM 보다 빠른 속도로 처리 가능하다

def split_x(seq, size):                          
    aaa = []                                        
    for i in range(len(seq) - size + 1):            
        subset = seq[i : (i+size)]                  
        aaa.append([j for j in subset])       
                                                   
    # print(type(aaa))                                
    return np.array(aaa)   

size = 6

# 1. 데이터
# npy 불러오기

samsung = np.load('./data/samsung.npy',  allow_pickle = True)
hite = np.load('./data/hite.npy', allow_pickle = True)

print('samsung_shape : ', samsung.shape) # (509, 1)
print('hite_shape : ', hite.shape)       # (509, 5)

########################중요! column이 1인 데이터는 미리 vector 형태로 만들어주자######################
samsung = samsung.reshape(samsung.shape[0], )
print('samsung.shape : ', samsung.shape) # (509, ) 으로 reshape 되었다

samsung = split_x(samsung, size)
print('samsung_shape(reshape) : ', samsung.shape) # (504, 6) # 5일치를 x, 다음 하루치를 y 따라서 6이 됨

# 삼성만 x, y 분리하면 됨
# 삼성 주가 예측하는 거니까 하이트는 y로 해 줄 필요가 없다(x로 쓰면 됨)

x_sam = samsung[:, 0:5] # 5일치의 데이터를 가지고
y_sam = samsung[:, 5]   # 6일째를 도출

print(x_sam)
print(y_sam)

print('x_sam.shape : ', x_sam.shape) # x_sam.shape :  (504, 5)
print('y_sam.shape : ', y_sam.shape) # y_sam.shape :  (504, )


# 현재 hite는 (509, 5), x_sam은 (504, 5)
# 열은 맞는데 행은 맞지 않다
# fit 하니까 에러메세지 발생
# ValueError: All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델은 행을 맞춰줘야 한다
# hite의 행을 잘라내는 방법밖에? x_hit로 가시오!
# 주식은 최근 것을 반영해주는 게 좋다
x_hit = hite[5:510, :] # 가장 예전의 5일치는 데이터에서 빼 준다
print('x_hit.shape : ', x_hit.shape) #  (504, 5)

# x_sam, x_hit의 행이 동일해졌다

# Sclaer 사용하기
scaler = MinMaxScaler() # 현재, x_sam 데이터는 2차원이므로 scaler 사용을 위해 따로 reshape 해 줄 필요가 없다
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)

scaler = MinMaxScaler()
scaler.fit(x_hit)
x_hit = scaler.transform(x_hit)


# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, ))
x1 = Dense(300)(input1)
x1 = Dropout(0.3)(x1)
x1 = Dense(700)(x1)
x1 = Dropout(0.3)(x1)
x1 = Dense(500)(x1)
x1 = Dropout(0.3)(x1)

input2 = Input(shape = (5, ))
x2 = Dense(200)(input2)
x1 = Dropout(0.3)(x1)
x2 = Dense(600)(x2)
x2 = Dropout(0.4)(x2)
x2 = Dense(400)(x2)
x1 = Dropout(0.3)(x1)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련

# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

modelpath = './test_samsung/model01/model01_check-{epoch:03d}-{loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_image = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

hist = model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 100, batch_size = 32, 
                callbacks = [cp], validation_split = 0.2, verbose = 1)

model.save('./test_samsung/model01/0602_test_model01_save.h5')

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 32)

print("loss : ", loss)
print("mse : ", mse)

x_sam_pred = x_sam[-1, :]
x_hit_pred = x_hit[-1, :]
x_sam_pred = x_sam_pred.reshape(-1, 5) # 새로 만든 x_sam_pred란 값을 다시 reshape 해줘야 y_pred 도출 시 차원 에러가 발생 안 한다
x_hit_pred = x_hit_pred.reshape(-1, 5)

y_pred = model.predict([x_sam_pred, x_hit_pred])
print("2020년 06월 02일 삼성전자의 예측 시가 : ", y_pred)



# 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['mse'], marker = '*', c = 'green', label = 'mse')
plt.plot(hist.history['val_mse'], marker = '*', c = 'purple', label = 'val_mse')
plt.grid()
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

# plt.show()

