# 몸풀기

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # 0부터 10까지 0.1씩 증가
y = np.sin(x) # 0.1에 대한 sin값, 0.2에 대한 sin값,,,,,

plt.plot(x, y) # sin(x) 그래프가 그려진다

plt.show()
