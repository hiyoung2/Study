# 과제 3) cnn 모델에서 parameter는 어떻게 계산되는가?
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape = (5, 5, 1)))     
model.add(Conv2D(5, (2, 2)))                                 
model.add(MaxPooling2D(pool_size = 2))
model.add(Dense(1))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 3, 3, 10)          100
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 5)           205
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 1, 1, 5)           0
_________________________________________________________________
dense_1 (Dense)              (None, 1, 1, 1)           6
=================================================================
'''
# CNN에서 파라미터는 '필터의 크기 * 입력채널의 수 * 출력채널의 수 + 출력채널의 bias' 로 계산 된다
# 첫 번째 레이어의 파라미터 계산
# (3 * 3) * 1 * 10 + 10 = 100
# 두 번째 레이어의 파라미터 계산
# (2 * 2) * 10 * 5 + 5 = 205

# 정정
# (input * kernel*kernel + bias) * output 로 하는 게 좋다
# cf) LSTM에서 4 * (input_dim + bias + output) * output


