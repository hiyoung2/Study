# 8.3 그래디언트 적용하기

# 함수 sum_of_squares 는 v가 0 벡터일 때 가장 작은 값을 가진다
# 만약 이 사실을 모른다고 가정, 경사 하강법을 이용해서 3차원 벡터의 최솟값을 구해보자
# 임의의 시작점을 잡고, 그래디언트가 아주 작아질 때까지 경사의 반대 방향으로 조금씩 이동하면 된다

'''
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float):
    # v에서 step_size만큼 이동하기
assert len(v) == len(gradient)
step = scalar_multiply(step_size, gradient)
return add(v, step)

def sum_of_squares_gradient(v: Vector):
    return [2 * v_i for in v]

# 임의의 시작점을 선택
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad - sum_of_squares_gradient(v)    # v의 그래디언트 게산
    v = gradient_step(v, grad, -0.01)    # 그래디언트의 음수만큼 이동
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001    # v는 0에 수렴해야 한다
'''


