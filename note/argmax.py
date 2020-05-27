import numpy as np
from keras.models import Sequential 
from keras.layers import LSTM, Dense

x = np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5]) 

from keras.utils.np_utils import to_categorical
y = to_categorical(y) 
y = y[:,1:] 
model = Sequential()
model.add(Dense(16, input_dim=(1), activation='relu'))
model.add(Dense(16, input_dim=(1), activation='relu'))
model.add(Dense(32, input_dim=(1), activation='relu')) 
model.add(Dense(64, input_dim=(1), activation='relu')) 
model.add(Dense(5, activation='softmax'))
model.summary()

#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=800, batch_size=1)  

a = model.predict([1,2,3,4,5])
print(np.argmax(a, axis = 1)+1)