
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten  # 각레이어마다 이미지를 자르기 때문에 역시 레이어 불러와야함

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape = (5, 5, 1)))     
# model.add(Conv2D(7, (3, 3)))                                 
# model.add(Conv2D(5, (2, 2)))               
# model.add(Conv2D(5, (2, 2)))                                 

model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten()) # 데이터를 쫙~ 펴 준다 / 3x3이 5장 겹쳐 있는 것을 하나씩 펼쳐진 느낌으로다가
model.add(Dense(1))

model.summary()