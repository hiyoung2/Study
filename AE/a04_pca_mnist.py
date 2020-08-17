#  copy 56, mnist auto encoder 적용

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


