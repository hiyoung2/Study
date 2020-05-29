import matplotlib.pyplot as plt

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

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
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2. 정규화
# x data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape)

# 2. 모델 구성
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=5, mode = 'auto') # mode 적지 않아도 auto로 적용됨
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
# Study / model에 파일이 생성된다 
# 파일명은 epoch : 훈련도, 02d : 에포를 두 자리 정수, val_loss : 4자리의 float, .4f : 소수 넷째자리까지, 파일명을 hdf라고 하겠다?

checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                             save_best_only=True, mode = 'auto') 
                            # save_best_only=True : 좋은 것만 저장하겠다
                            # val loss monitor하고 최고로 좋은 값을 저장해서 파일을 만들겠다
                            # checkpoint도 모드가 있음 (안 적어도 되는데 그건 또 디폴트가 있다는 것)
                            # modelpath : 변수명, 위에 위치를 대입해 놓은
                            # 파일경로 filepath : 모델 path
                            # 01 : epoch, 0.0739 : loss (model 폴더에서 생성 되고 있음)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=200, callbacks = [early_stopping, checkpoint], 
                                   validation_split = 0.2, verbose = 1) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

plt.figure(figsize = (10, 6))

plt.subplot(2, 1, 1) # 2, 1 : 2행 1열의 그림을 그리겠다, 마지막 1은 : 2행 1열의 첫 번째 것을 그리겠다
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
plt.grid() # 모눈종이처럼 그림에 가로 세로 줄이 그어져 나옴
plt.title('loss')      
plt.ylabel('loss')      
plt.xlabel('epoch')          
# plt.legend(['loss', 'val_loss']) 
plt.legend(loc = 'upper right') # loc : location, 우측 상단

plt.subplot(2, 1, 2) # 2행 1열의 2번째 그림
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() 
plt.title('acc')      
plt.ylabel('acc')      
plt.xlabel('epoch')          
plt.legend(['acc', 'val_acc']) 
plt.show()  
