# 53 copy , 만들어 놓은 모델에 dropout 을 적용시켜서 성능을 비교해보자

# dropout : 딥러닝에서 overfitting을 줄이는 방법, 탈락, 낙오라는 뜻의 dropout은
# 전체 weight를 게산에 참여시키는 게 아니라 레이어에 포함 된 weight 중에서 일부만 참여시킨다
# 적용 시키기 전과 후를 비교하면 dropout layer를 적용한 게 더 좋은 성능을 보여준다(고 한다,, 실습중)
# 일단 모델을 짤 때 데이터 크기에 맞게 잘 짜야한다
# 지금 데이터가 60000개로 엄청난데 노드 수를 10단위로 한다면 dropout을 적용해도 의미가 없다고 봐야한다
# Dropout은 결국 노드 수를 줄이는 것과 같다, 직접 하나씩 수정하기보다 일정 비율을 정해서 노드 수를 줄이는 거니까
# 애초에 노드 수가 적으면 효과가 없다
# 데이터 크기 생각하지 못하고 노드 수를 적게 잡았더니 acc가 영 나오지 않았음

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist 

# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print('x_train : ', x_train[0])
print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )

print(x_train[0].shape)
# print(y_train[0])
# plt.imshow(x_train[0], 'gray') 
# plt.imshow(x_train[0]
# plt.show()

# 데이터 전처리 1. 원핫인코딩
# y data
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2. 정규화
# x data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(77, (2, 2), input_shape = (28, 28, 1)))     
model.add(Conv2D(111, (3, 3), activation = 'relu'))
model.add(Dropout(0.2))     

model.add(Conv2D(99, (3, 3), padding = 'same'))   
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))          

model.add(Conv2D(55, (2, 2), padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

# 3. compile, 훈련
# from keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor='loss', patience=5, mode = 'auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=70, batch_size=200, validation_split = 0.2) 

# 4. 예측, 평가

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)

print('loss : ', loss)
print('acc : ' , acc)

# x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_test)


# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)

# acc 0.982로 올려라,,,,,



'''
model = Sequential()
model.add(Conv2D(77, (2, 2), input_shape = (28, 28, 1)))     
model.add(Conv2D(111, (3, 3), activation = 'relu'))
model.add(Dropout(0.2))     

model.add(Conv2D(55, (3, 3)))   
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))          

model.add(Conv2D(33, (2, 2), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

epoch = 80, batch_size 100

acc :  0.9923999905586243
'''
'''
epoch 100
acc :  0.9919999837875366
'''
'''
epoch 70
acc :  0.9923999905586243
'''
'''
epoch 60
acc :  0.9922000169754028
'''
'''
model = Sequential()
model.add(Conv2D(77, (2, 2), input_shape = (28, 28, 1)))     
model.add(Conv2D(111, (3, 3), activation = 'relu'))
model.add(Dropout(0.2))     

model.add(Conv2D(99, (3, 3)))   
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))          

model.add(Conv2D(55, (2, 2), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
epoch 70, batch_size 200 
acc :  0.9930999875068665
'''
