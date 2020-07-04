# leakyrelu visualization

import numpy as np
import matplotlib.pyplot as plt

# def leaky_relu(x) :
#     return np.maximum(1.5 * x, x)
#     # 0.15 자리 = 알파값, 튜닝 가능
#     # import 해 주고 나서 튜닝하려면 복잡해진다?

def leaky_relu(x, a = 0.01) :
    return np.maximum(a*x, x)

# x > 0.01*x -> x == x
# x < 0.01*x -> x == 0.01x


x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()