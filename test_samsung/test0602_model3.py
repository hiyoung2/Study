# LSTM 2개 구현
# concatenate를 Concatenate로 변경해봐라
# concatenate ([x1, x2]) / Concatenate()([x1, x2])
# 문법에 차이가 있다

import numpy as np
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

# samsung data vecotr 형태로 만들어주기
samsung = samsung.reshape(samsung.shape[0], )
print('samsung.shape : ', samsung.shape) # (509,)



samsung = split_x(samsung, size)
print('samsung_shape : ', samsung.shape) 
                                                    
x_sam = samsung[:, 0:5] 
y_sam = samsung[:, 5] 

print('x_sam.shape : ', x_sam.shape) # (504, 5)
print('y_sam.shape : ', y_sam.shape) # (504, )
print('hite.shape : ', hite.shape)   # (509, 5)

# hite를 x_sam과 행 맞춰주기
x_hit = hite[5:510, 0:4] # 거래량은 제거했음 그리고 행 맞춰주기 위해서 hite data의 가장 오래된 5일치 제거
print('x_hit.shape : ', x_hit.shape) #  (504, 4)
print(x_hit)

# scaler
scaler = MinMaxScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)

scaler = MinMaxScaler()
scaler.fit(x_hit)
x_hit = scaler.transform(x_hit)


x_sam = x_sam.reshape(x_sam.shape[0], x_sam.shape[1], 1)
x_hit = x_hit.reshape(x_hit.shape[0], x_hit.shape[1], 1)

print('x_sam.shape : ', x_sam.shape) # x_sam.shape :  (504, 5, 1)
print('x_hit.shape : ', x_hit.shape) # x_hit.shape :  (504, 5, 1)

# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, 1))
x1 = LSTM(100)(input1)
x1 = Dense(120)(x1)
x1 = Dense(150)(x1)
x1 = Dense(90)(x1)

input2 = Input(shape = (4, 1))
x2 = LSTM(100)(input2)
x2 = Dense(150)(x2)
x2 = Dense(200)(x2)
x2 = Dense(50)(x2)

merge = Concatenate()([x1, x2]) # concatenate ([x1, x2]) / Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 32)

print("loss : ", loss)
print("mse : ", mse)

x_sam_pred = x_sam[-1, :, :]
x_hit_pred = x_hit[-1, :, :]

# print(x_sam_pred.shape) # (5, 1)

# y_pred = model.predict([x_sam_pred, x_hit_pred])
# print(y_pred)
# ValueError: Error when checking input: expected input_1 to have 3 dimensions, but got array with shape (5, 1)
# x_sam_pred.shape을 3차원으로 다시 맞춰주면 해결된다

x_sam_pred = x_sam_pred.reshape(-1, 5, 1)
x_hit_pred = x_hit_pred.reshape(-1, 4, 1)

print('x_sam_pred.shape : ', x_sam_pred.shape) # (1, 5, 1)

y_pred = model.predict([x_sam_pred, x_hit_pred])
print(y_pred)