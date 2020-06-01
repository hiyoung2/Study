# 과제 1
# Sequential형으로 작성하시오

import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

# 1. 데이터 준비

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape)   # (60000, 28, 28)
print("x_test.shape : ", x_test.shape)     # (10000, 28, 28)
print("y_train.shape : ", y_train.shape)   # (60000,)
print("y_test.shape : ", y_test.shape)     # (10000,)

# 데이터 전처리
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0


# 2. 모델 구성 # Dropout 적극활용하기, 직접 노드 수 줄이기보다 편리

model = Sequential()

model.add(Conv2D(88, (2, 2), input_shape = (28, 28, 1)))     
model.add(Dropout(0.2))  
model.add(Conv2D(111, (3, 3), activation = 'relu'))
model.add(Dropout(0.3))     

model.add(Conv2D(99, (3, 3), padding = 'same'))   
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(66, (2, 2), padding = 'same',activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

# model.save('./model/sample/fashionmnist/fashionmnist_model_save.h5') 
# 모델 구성 후, fit 전에 저장 -> 모델 자체만 저장이 된다

# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
checkpointpath = './model/sample/fashionmnist/fashionmnist_check-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = checkpointpath, monitor = 'val_loss', 
                             save_best_only = True, save_weights_only = False, verbose = 1)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 95, batch_size = 100, callbacks = [checkpoint], validation_split = 0.3, verbose = 1)

# model.save_weights('./model/sample/fashionmnist/fashionmnist_save_weight.h5')

# model.save('./model/sample/fashionmnist/fashionmnist_checkpoint_best.h5')

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)

# print(y_pred)
print(np.argmax(y_pred, axis = 1))
# print(y_pred.shape)


