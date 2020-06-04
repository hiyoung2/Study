# LSTM과 DENSE를 앙상블로 만들어보자

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

samsung = samsung.reshape(509, ) # 일단, samsung 데이터를 벡터화 시켜주자

samsung = split_x(samsung, size)
print('samsung_shape : ', samsung.shape) # (504, 6)

x_sam = samsung[:, 0:5] 
y_sam = samsung[:, 5] 

print('x_sam.shape : ', x_sam.shape) # (504, 5)
print('y_sam.shape : ', y_sam.shape) # (504,)


# 행 맞춰주기
x_hit = hite[5:510, 0:4] # 거래량은 제거
print('x_hit.shape : ', x_hit.shape) #  (504, 4)
print(x_hit)



# Scaler(2차원으로 만들어줘야 한다)
scaler = MinMaxScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)

scaler = MinMaxScaler()       # x_hit 데이터는 2차원이니까 따로 reshape 안 해줘도 됨
scaler.fit(x_hit)
x_hit = scaler.transform(x_hit)

x_sam = x_sam.reshape(504, 5, 1) # LSTM model에 넣을 것이기 때문에 다시 3차원으로 reshape 해 준다

print('x_sam.shape : ', x_sam.shape) # (504, 5, 1)
print('x_hit.shape : ', x_hit.shape) # (504, 4)


# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, 1))
x1 = LSTM(100)(input1)
x1 = Dense(300)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(500)(x1)
x1 = Dropout(0.4)(x1)
x1 = Dense(300)(x1)

input2 = Input(shape = (4, ))
x2 = Dense(90)(input2)
x2 = Dense(400)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(200)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(50)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

modelpath = './test_samsung/model02/model02_check-{epoch:03d}-{loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_image = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 100, batch_size = 16, 
                 callbacks = [cp], validation_split = 0.2, verbose = 1)

model.save('./test_samsung/model02/0602_test_model02_save.h5')

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 16)

print("loss : ", loss)
print("mse : ", mse)


# predict 
x_sam_pred = x_sam[-1, :, :]
x_hit_pred = x_hit[-1, :]

x_sam_pred = x_sam_pred.reshape(-1,5,1) # pred 값을 잡아줬는데 y_pred에 넣으면 자꾸 차원이 맞지 않다고 떠서 다시 reshape 해주니까 잘 나옴
x_hit_pred = x_hit_pred.reshape(-1,4)

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
