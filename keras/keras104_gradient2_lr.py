import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
y = f(x)

# 시각화
plt.plot(x, y, 'k-')
# - solid line

plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

f = lambda x : x**2 - 4*x + 6
gradient = lambda x : 2*x - 4

# f를 미분 -> graident
# f의 기울기가 0이 되는 지점을 찾아보자
# 즉, gradient가 0이 되는 지점을 찾는 과정이 된다

x0 = 0.0
MaxIter = 10
lr = 0.25
# 딥러닝에서는 lr, learning rate(학습률)의 loss를 최소화
# optimizer가 필요하므로 model.compile에 있다

print("step\t x\t f(x)")
print("{:02d}\t {:6.5f}\t {:6.5f}".format(0, x0, f(x0)))

# 경사하강법 나타내는 코드
for i in range(MaxIter) :
    x1 = x0 - lr*gradient(x0) 
    x0 = x1

    print("{:02d}\t {:6.5f}\t {:6.5f}".format(i+1, x0, f(x0)))


'''
step     x       f(x)
00       0.00000         6.00000
01       1.00000         3.00000
02       1.50000         2.25000
03       1.75000         2.06250
04       1.87500         2.01562
05       1.93750         2.00391
06       1.96875         2.00098
07       1.98438         2.00024
08       1.99219         2.00006
09       1.99609         2.00002
10       1.99805         2.00000
'''