import numpy as np 

x = np.array([1,2,3,4,5,6,7,8,9,10])    
y = np.array([1,2,3,4,5,6,7,8,9,10])

from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import EarlyStopping, TensorBoard # TemsprBoard 불러오기 완료
# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

model = Sequential()   
model.add(Dense(5, input_dim=1, activation='relu')) 
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs= 100, batch_size=1) 

'''
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)
'''


