# relu visualization

import numpy as np
import matplotlib.pyplot as plt

def relu(x) :
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# y값이 0 이하면 무조건 0으로 수렴, 0을 넘어서면 선형으로 증가

# 많이 쓰는 것?
# relu > leakyrelu > elu > selu