# mnist

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist # keras.datasets : 케라스의 데이터셋, 예제파일이 들어가 있는 것들(케라스에서 제공해 줌, 와우, 깔끔한 예제들이라고 함)

(x_train, y_train), (x_test, y_test) = mnist.load_data() # mnist.load_dat가 x_train y_train x_text y_test 로 반환, 트레인과 테스트 자동 분리해 준다

print('x_train : ', x_train[0])
print('y_train : ', y_train[0]) # 예제용 데이터 다운로드 실행 된다, 한 번만 다운 받으면 됨


print('x_train.shape : ', x_train.shape) # (60000, 28, 28)
print('x_test.shape : ', x_test.shape)   # (10000, 28, 28)
print('y_train.shape : ', y_train.shape) # (60000, )
print('y_test.shape : ', y_test.shape)   # (10000, )


print(x_train[1].shape)
# print(y_train[0])
plt.imshow(x_train[59999], 'gray') #imshow : 이미지를 보여줌 
# plt.imshow(x_train[0]
plt.show()



