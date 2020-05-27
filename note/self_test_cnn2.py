from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (4, 4), input_shape = (20, 20, 1)))     
model.add(Conv2D(6, (2, 2)))                                 
model.add(Conv2D(2, (2, 2)))               
model.add(Conv2D(5, (4, 4)))                                 

model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(1))

model.summary()