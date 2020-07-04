# tanh visualization

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

# 탄젠트는 넘파이에서 제공하기 때문에 함수 만들지 않고 바로 쓸 수 있음

plt.plot(x, y)
plt.grid()
plt.show()
# y값이 -1 ~ 1 사이에 수렴