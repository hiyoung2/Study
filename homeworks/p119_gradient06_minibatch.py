# 앞서 살펴본 예시 미니배치로 다시 푸기
'''
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

import random
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grade = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1,        "slope should be about 20"
assert 4.9 < intercept < 5.1,      "intercept should be aoubt 5"

# SGD(stochastic gradient descent)의 경우에는 각 에폭마다 단 하나의 데이터 포이늩에서 그래디언트 계산
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slpe, intercept = theta

assert 19.9 < slope <20.1,      "slope should be about 20"
assert 4.9 < intercept < 5.1,   "intercept should be about 5"
'''