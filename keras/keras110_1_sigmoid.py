# sigmoid visualization

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1/(1+np.exp(-x))

    # np.exp : 지수함수
    # 밑이 자연상수 e인 지수함수로 볂놘
    # e^x


x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print("x.shape : ", x.shape)
print("y.shpae : ", y.shape)


plt.plot(x, y)
plt.grid()
plt.show()

# sigmoid 함수 그래프
# y 값이 0 ~ 1 사이에 수렴
# activation,  활성화 함수 의 목적 : 가중치 값을 한정시킨다?