import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

samsung = pd.read_csv("./data/csv/samsung.csv",
                  index_col = 0, # 첫 번째 열(일자)를 인덱스로 처리(실제 데이터가 아니니까)
                                 # index_col = None을 해 버리면 인덱스가 없다는 뜻이 된다
                  header = 0,    # 시가 등이 적힌 첫 번째 행을 헤더로 처리(실제 데이터가 아니니까)
                  sep = ',',     # 
                  encoding = 'CP949') # 

hite = pd.read_csv("./data/csv/hite.csv",
                  index_col = 0, 
                  header = 0, 
                  sep = ',',
                  encoding = 'CP949')

print(samsung.head())
print(hite.head())
print('samsung_shape : ', samsung.shape) # (700, 1)
print('hite_shape : ', hite.shape)       # (720, 5)

# 결측치가 들어간 데이터로 준비되었음, 의도치 않은 결측치 제거 공부
# 단순하게 슬라이싱 하면 된다?
# hite data의 6월 2일자에는 시가는 적혀 있는데 나머지는 Nan이라 어떻게 해결해야 할지 몰랐음
# 문자열로 이루어진 데이터를 숫자형으로 바꾸는 것도 많이 힘들었음

# 앙상블 -> 단순 더하기, 가중치가 50 대 50
# 고가, 저가, 종가, 거래량 없애고 시가만 넣는 방법도 있다
# 이렇게 하면 결측치 제거할 필요도 없음
# 그래도 다 넣어서 해 보자
# 삼성은 시가 하나만 있음, 결측치만 제거하면 끝

# 슬라이싱을 하면 몇 번째 인덱스부터인지 찾아봐야함
# 양이 많은 데이터는 그렇게 하기가 힘듦
# 다른 방식으로 NAN을 제거해보자

# NA : Not Available, NAN : Not A Number

# Nan 제거 1
samsung = samsung.dropna(axis = 0) 

# dropna : Nan 제거
# axis = 0(index), 행을 따라 동작한다 
# axis = 1(columns), 열을 따라 동작한다

print('samsung_shape ; ', samsung.shape)
print(samsung)

hite = hite.fillna(method = 'bfill')
hite = hite.dropna(axis = 0) # hite data에서 Nan이 들어있는 행을 제거한다
print(hite)

# bfill : 그 전날값으로 채워진다

# 결측값을 특정 값으로 채우는 방법들 : df.fillna()
# 1. 0으로 채우기 :  df.fillna(0)
# 2. 문자열로 채우기 : df.fillna('missing') : 'missing' 이라는 string 값으로 채워진다
# 3. 결측값을 앞 방향으로 채우기 : fillna(method = 'ffill') or fillna(method = 'pad')
# 4. 뒷 방향으로 채우기 : fillna(method = 'bfill') or filnna(method = 'backfill')
# 5. 앞/뒤 방향으로 결측값 채우는 횟수를 제한하기 
# - fillna(method = 'ffill', limit = number)
# - filnna(method = 'bfill', limit = number)
# 6. 결측값을 변수별 평균으로 대체하기
# - df.fillna(df.mean()) or
# - df.where(pd.notnull(df), df.mean(), axis = 'colunms')

# 현재 별 차이가 없으므로 그냥 전날값(뒷방향에서 가져옴, 현재 앞방향엔 값이 없음)을 채워도 된다

# Nan 제거 2
# hite = hite[0:509]

## iloc로 하는 방법
# hite.iloc[0, 1:5] = ['10', '20', '30', '40'] # non의 위치가 첫 행의 1, 2, 3, 4 열에 있었
#                                      그 자리에 10, 20, 30, 40으로 대신 채워줌
#                                      iloc에서 i는 index, iloc와 loc를 알고 있어야 함
# print(hite)

## loc로 하는 방법
# hite.loc["2020-06-02", "고가" : "거래량"] = ["10", "20", "30", "40"]
# print(hite)

# Nan 부분을 predict로 넣어서 하는 방법도 있음
# 그러면 모델을 하나 더 짜줘야 하는데
# 머신러닝에선 간단한 방식으로 해결할 수 있다

# 증권사 데이터는 항상 최신 날짜가 위에 있으므로 다시 정렬을 해 줄 필요가 있다
# 숫자인 줄 알았는데 문자 + 콤마까지 들어 있다
# 정렬, 형변환 문제를 해결해야 한다

# 현재 내림차순 상태(최근 데이터값이 상위), 오름차순으로 바꿔줘야 한다

samsung = samsung.sort_values(['일자'], ascending = [True]) # 오름차순 = True / False이면 내림차순으로(디센딩)
hite = hite.sort_values(['일자'], ascending = [True])

print(samsung)
print(hite)

# 콤마제거, 문자를 정수로 형변환

for i in range(len(samsung)) : # '37,000' = string
    samsung.iloc[i,0] = int(samsung.iloc[i, 0].replace(',', ''))  # 37000 (콤마 사라지고 int형 변환됨)
    # iloc : index location
    # (0,0) 자리에 콤마를 제거하겠다
print(samsung)

print(type(samsung.iloc[0,0])) # <class 'int'>

for i in range(len(hite)) : #len(hite) : 509
    for j in range(len(hite.iloc[i])) : # 각 행을 열의 갯수만큼 돌림(현재 5번)
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',', ''))

# 509의 행마다 5개의 열을 다 변환을 바꿔준다, for문을 두 개 써줘야 한다
# 행, 열 모두 바꿔줘야하니까


print(hite)
print(type(hite.iloc[1,1])) # <class 'int'>

print('samsung_shape : ', samsung.shape) # (509, 1) 509행 1열 != (509, ) -> 이거는 509개의 스칼라, 하나의 벡터
print('hite_shape : ', hite.shape)       # (509, 5) 509행 5열 

saumsung = samsung.values
hite = hite.values

print(type(hite)) # <class 'numpy.ndarray'>

# npy 파일로 저장 완료
np.save('./data/samsung.npy', arr = samsung)
np.save('./data/hite.npy', arr = hite)



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

samsung = samsung.reshape(samsung.shape[0], ) 
print('samsung_vec.shape : ', samsung.shape) # (509, )
# samsung data, split 함수로 나눠주기


samsung = split_x(samsung, size)
print('samsung_shape : ', samsung.shape) # (504, 6)


x_sam = samsung[:, 0:5] 
y_sam = samsung[:, 5] 

print('x_sam_split.shape : ', x_sam.shape) # (504, 5)
print('y_sam_split.shape : ', y_sam.shape) # (504,)



# scaler 먼저 무조건 써야 한다, 스케일러 다음에 pca 써야 함
x_sam = x_sam.reshape (504, 5)

scaler = MinMaxScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)

scaler = MinMaxScaler()
scaler.fit(hite)
hite = scaler.transform(hite)

x_sam = x_sam.reshape(-1, 5, 1)
x_hite = hite.reshape(-1, 5, 1)

# 차원 축소,  PCA
pca = PCA(n_components = 1)
pca.fit(hite)
x_hit = pca.transform(hite)


print('x_sam.shape : ', x_sam.shape) # (504, 5, 1)
print('y_sam.shape : ', y_sam.shape) # (504, 1)
print('x_hit.shape : ', x_hit.shape) # (509, 1)


# 가장 마지막에 hite를 split 함수에 넣어준다
x_hit = split_x(x_hit, size)
print('x_hit_shape : ', x_hit.shape) # (504, 6, 1)

# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, 1))
x1 = LSTM(100)(input1)
x1 = Dense(300)(x1)
x1 = Dense(350)(x1)
x1 = Dense(200)(x1)  

input2 = Input(shape = (6, 1))
x2 = LSTM(100)(input2)
x2 = Dense(300)(x2)
x2 = Dense(350)(x2)
x2 = Dense(200)(x2)

merge = Concatenate()([x1, x2]) # concatenate ([x1, x2]) / Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련

modelpath = './test/test0602_hiy_check-{epoch:03d}-{loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss', save_best_only = True, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_image = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 200, batch_size = 16, 
                  callbacks = [cp], validation_split = 0.2, verbose = 1)


model.save('./test/test0602_hiy_save.h5')

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 16)

print("loss : ", loss)
print("mse : ", mse)

# predict 값 잡아주기(가장 최근의 것으로 가져옴)
x_sam_pred = x_sam[-1, :, :]
x_hit_pred = x_hit[-1, :, :]
x_sam_pred = x_sam_pred.reshape(-1,5,1) 
x_hit_pred = x_hit_pred.reshape(-1,6,1)


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
