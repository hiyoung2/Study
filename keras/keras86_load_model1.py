# keras54 복붙
import numpy as np
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

# print(x_train.shape)

# 2.  모델 구성(불러오기)
from keras.models import load_model
model = load_model('./model/model_test01.h5')

model.summary()


# 3. 컴파일, 훈련

## 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

print('loss : ', loss)
print('acc : ' , acc)

# y_pred = model.predict(x_test)
# print(y_pred)
# print(np.argmax(y_pred, axis = 1))


# 시각화 
# plt.figure(figsize = (10, 6)) 

# plt.subplot(2, 1, 1)
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')         
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')   
# plt.grid() 
# plt.title('loss')      
# plt.ylabel('loss')      
# plt.xlabel('epoch')          
# # plt.legend(['loss', 'val_loss']) 
# plt.legend(loc = 'upper right')

# plt.subplot(2, 1, 2)
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid() 
# plt.title('acc')      
# plt.ylabel('acc')      
# plt.xlabel('epoch')          
# plt.legend(['acc', 'val_acc']) 
# plt.show()  


'''
keras85_save1.py 의 fit한 결과와 같다
-> 가중치가 저장이 되었다!
loss :  0.030102645093202592
acc :  0.9897000193595886
'''

'''
모델 구성 한 다음 저장한 걸 불러오니까
fit한 건 저장이 안 되기 때문에 compile 오류가 뜸
RuntimeError: You must compile a model before training/testing. Use `model.compile(optimizer, loss)`.
'''

'''
저장한 모델을 불러와서 내가 따로 튜닝 없이 compile, fit을 하려면
모델을 저장할 시에 fit 과정 다음에 저장을 해줘야 한다
'''