import numpy as np  
import matplotlib.pyplot as plt

from keras.datasets import cifar100 # column 100개 짜리 예제 ㅎㄷㄷ(현존하는 가장 큰 놈)
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D, Dense, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# 1. 데이터 준비

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(y_test.shape)  # (10000, 1)

# 1.1 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 1.2 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape)

# 2. 모델 구성

input1 = Input(shape = (32, 32, 3))
dense1 = Conv2D(50, (2, 2))(input1)
dense2 = Dropout(0.2)(dense1)     
dense3 = Conv2D(100, (3, 3), activation = 'relu')(dense2)
dense4 = Dropout(0.2)(dense3)     

dense5 = Conv2D(150, (3, 3) , padding = 'same', activation = 'relu')(dense4)   
dense6 = MaxPooling2D(pool_size = 2)(dense5)
dense7 = Dropout(0.3)(dense6)          

dense8 = Conv2D(30, (2, 2), padding = 'same', activation = 'relu')(dense7)
dense9 = MaxPooling2D(pool_size = 2)(dense8)
dense10 = Dropout(0.1)(dense9)

dense11 = Flatten()(dense10)
output1 = Dense(100, activation = 'softmax')(dense11)

model = Model(inputs = input1, outputs = output1) 

model.summary()

# model.save('./model/sample/cifar100/cifar100_model_save.h5') 


# 3. 컴파일, 훈련
# early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')

# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq=0, write_graph = True, write_images = True)

checkpointpath = './model/sample/cifar100/cifar100_check-{epoch:02d}-{loss:.4f}.hdf5'

checkpoint = ModelCheckpoint(filepath = checkpointpath, monitor = 'loss', 
                             save_best_only=True) 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 30, batch_size = 100,  callbacks = [checkpoint], verbose = 1)

model.save_weights('./model/sample/cifar100/cifar100_save_weight.h5')

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)
print('loss : ', loss)
print('acc : ', acc)

# 시각화

# plt.figure(figsize = (10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.subplot(2, 1, 2)
# plt.plot(hist.history['acc'], marker = '.', c = 'green', label = 'acc')
# plt.plot(hist.history['val_acc'], marker = '.', c = 'purple', label = 'val_acc')
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')
# plt.show()

