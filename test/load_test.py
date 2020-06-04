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
                  index_col = 0, 
                                 
                  header = 0,    
                  sep = ',',      
                  encoding = 'CP949')  

hite = pd.read_csv("./data/csv/hite.csv",
                  index_col = 0, 
                  header = 0, 
                  sep = ',',
                  encoding = 'CP949')

print(samsung.head())
print(hite.head())
print('samsung_shape : ', samsung.shape) # (700, 1)
print('hite_shape : ', hite.shape)       # (720, 5)

samsung = samsung.dropna(axis = 0) 



print('samsung_shape ; ', samsung.shape)
print(samsung)

hite = hite.fillna(method = 'bfill')
hite = hite.dropna(axis = 0) #
print(hite)

samsung = samsung.sort_values(['일자'], ascending = [True])
hite = hite.sort_values(['일자'], ascending = [True])

print(samsung)
print(hite)

for i in range(len(samsung)) :
    samsung.iloc[i,0] = int(samsung.iloc[i, 0].replace(',', ''))

print(samsung)

print(type(samsung.iloc[0,0])) 

for i in range(len(hite)) : 
    for j in range(len(hite.iloc[i])) :
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',', ''))


print(hite)
print(type(hite.iloc[1,1]))

print('samsung_shape : ', samsung.shape) 
print('hite_shape : ', hite.shape)      

saumsung = samsung.values
hite = hite.values

print(type(hite)) 

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

# model load
from keras.models import load_model
model = load_model('./test/test0602_hiy_save.h5')


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


