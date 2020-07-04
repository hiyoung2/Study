# loss 함수 그래프 식으로 나타내면 y = ax^2 + bx + c
# y = wx + b 를 구하는 과정 : 최적의 가중치, 최소의 loss(cost)
# 미분하여 최적의 가중치 구하기
# learning rate 작게 잡으면 그라디언트 손실이 발생
# learning rate 크게 잡으면 그라디언트 폭주가 발생
# Vanishing Gradient problem : 기울기 소실 문제, 기울기값이 사라지는 문제

# lambda : 간략한 함수
# return 없이 바로 들어간다


# 1. lambda 사용
gradient = lambda x: 2*x - 4
# x^2 - 4x + b 를 적분한 것 : 2x - 4

# 2. definition 사용
def gradient2(x) :
    temp = 2*x - 4 
    return temp


x = 3

print(gradient(x))  # 2
print(gradient2(x)) # 2