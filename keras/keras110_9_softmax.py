import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
print(x)    
# [1 2 3 4]
y = softmax(x)

ratio = y
labels = y

# startangle : 첫 번째 pie의 시작 각도
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()
# 각 레이블에 있는 숫자를 다 더하면 1