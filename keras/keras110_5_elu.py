# elu visualization

import numpy as np
import matplotlib.pyplot as plt


# 방법 1
def elu(x) :
    x = np.copy(x)
    
    # copy : 전혀 다른 메모리 공간에 값만 같은 배열을 복사
    # copy 없이 하면 기존의 x값이 변경된다

    x[x<0] = 0.2*(np.exp(x[x<0]-1))
    return x


x = np.arange(-5, 5, 0.1)
y = elu(x)

# 방법 2
# list comprehension
# a = 0.2
# x = np.arange(-5, 5, 0.1)
# y = [x if x> 0 else a * (np.exp(x)-1) for x in x]


'''
def elu(x) :
    if (x > 0) :
        return x
    if (x < 0) :
        return 0.2*(np.exp(-x)-1)
# 0.2 = 알파값

def elu(x) :
    y_list = []
    for x in x :
        if (x > 0) :
            y = x
        if (x < 0) :
            y = 0.2*(np.exp(x) - 1)
        y_list.append(y)
    return y_list
'''


plt.plot(x, y)
plt.grid()
plt.show()