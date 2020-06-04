# 8.2 그래디언트 계산하기
'''
from typiing import Callable

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float):
    return (f(x + h) - f(x)) / h


# 도함수(derivative) 구하기
# 다음의 square 함수는 
def squre(x: float):

# 이러한 도함수를 갖는다
def derivative(x: float):;
return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(squre, x, h = 0.001) for x in xs]

# 두 계산식의 결괏값이 거의 비슷함을 보여주기 위한 그래프
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label = 'Actual')   # 빨간색 x
plt.plot(xs, estimates, 'b+', label = 'Estimate') # 파란색 +
plt.legend(loc = 9)
plt.show()

# 만약 f가 다변수 함수라면 여러 개의 입력 변수 중 하나에 작은 변화가 있을 때
# f(x)의 변화량을 알려주는 편도함수(partial derivative) 역시 여러 개 존재

# i 번째 편도함수는 i 번째 변수를 제외한 다른 모든 입력 변수를 고정시켜서 계산할 수 있다
def partial_difference_quotient(f: Callable[[Vector], float], 
                                v: Vector,
                                i: int,
                                h: float):
    # 함수 f의 i번째 편도함수가 v에서 가지는 값
    w = [v_j + (h if j == i else 0)
        for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

# 그 다음에는 일반적인 도함수와 같은 방법으로 그래디언트의 근삿값을 구할 수 있다
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_puotient(f, v, i, h)
           for i in range(len(v))]
'''