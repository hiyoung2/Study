# keras 56번 파일 pca 적용시켜서 코드 완성

import numpy as np

from tensorflow.keras.datasets import mnist 
from sklearn.decomposition import PCA
# 1. 데이터 준비 (mnist에서 불러왔다 , 가로세로 28짜리)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# print('x_train : ', x_train[0])
# print('y_train : ', y_train[0])

print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )

# print(x_train[0].shape)
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

# # 데이터 전처리 - 정규화
# x data
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

X = np.append(x_train, x_test, axis = 0) # 두 가지를 붙여 버린다(pca 함께 적용시키기 위해, append 사용)
print("X.shape :", X.shape) # X.shape : (70000, 784)

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

best_n_components = np.argmax(cumsum >= 0.95) + 1
print(best_n_components) # 154


# pca = PCA(n_components = 5)
# X_pca = pca.fit_transform(X)

pca = PCA(n_components = 154)
X_pca = pca.fit_transform(X)



# from sklearn.model_selection import train_test_split
# x_train, x_test = train_test_split(X, train_size = 0.85)

x_train = X_pca[:60000, :]
x_test = X_pca[60000:, :]
print('x_train.shape :', x_train.shape) # (60000, 154)
print('x_test.shape :', x_test.shape) # (10000, 154)


# 2. 모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

input_img = Input(shape = (154, ))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(512, activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dense(10, activation = 'softmax'))

layer = Dense(256, activation = 'relu')(input_img)
layer = Dropout(0.3)(layer)
layer = Dense(512, activation = 'relu')(layer)
layer = Dropout(0.4)(layer)
layer = Dense(256, activation = 'relu')(layer)
layer = Dropout(0.3)(layer)
output_img = Dense(10, activation = 'softmax')(layer)

model = Model(inputs = input_img, outputs = output_img)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split = 0.2) 

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ' , acc)

y_pred = model.predict(x_test)

# print(y_pred)
print(np.argmax(y_pred, axis = 1))
print(y_pred.shape)


# epo 200, batch 256
# loss :  0.0862143378296867
# acc :  0.9846

# epo 100, batch 200
# loss :  0.08644788220924277
# acc :  0.9826