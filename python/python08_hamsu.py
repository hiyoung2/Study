# model을 함수 x, 함수형!으로 만듦 (함수 동생 아니고)
# 앞선 수업에서 만든 앙상블 - 모델 5개를 각각의 함수로 만들 수 있다, 재사용 가능

# class
# 우리 반 : class
# 학생, 모니터, 학생들의 성적, 학생들의 능력치 등등 ( -> 변수) 이 class에 들어 있다

# model 짜다가 class에 대해 나오면 다시 배울 것

# 현재는 함수까지만 알고 있으면 된다

# def sum(a, b):                  # sum을 son으로 하든 young으로 하든 이름 아무거나 정의 가능 
#     return a+b                  # but, 다른 사람들도 이해할 수 있게 하기 위해 통상적인 이름으로 정하자
#                                 # def sum(a, b) a와 b 라는 변수 2개를 받겠다(매개변수)
#                                 # return 에다가 위에 쓴 변수 이름 그대로 써야 함
#                                 # return a와 b가 아닌 예를 들어 c+d 로 하면 안 됨 (c+d를 하려면 매개변수에 c, d로 해줘야함)
#                                 # a와 b라는 변수를 받아들여 a + b라는 연산을 하고 그 값을 sum 함수에 반환해준다
# print(sum1(3, 4))

# def sum1(c, d):                 
#     return c + d

# a = 1
# b = 2
# c = sum1(a, b)                

# print(c)                        # 함수명 ( , ) 각각의 위치만 맞춰주면 된다

def sum1(a, b):                 
    return a + b

a = 1
b = 2
c = sum1(a, b)
print(c)

############## 곱셈, 나눗셈, 뺄셈도 만들어보자
# 함수명 : mul1, div1, sub1

print("=================곱셈====================")

def mul1(a, b):
    return a * b

a = 1
b = 2
c = mul1(a, b)
print(c)

print("=================나눗셈==================")

def div1(a, b):
    return a / b

a = 6
b = 3
c = div1(a, b)
print(c)

print("=================뺄셈====================")

def sub1(a, b):
    return a - b

a = 10
b = 7
c = sub1(a, b)
print(c)

# 함수의 목적 : 재사용

print("============매개변수가없는함수===========")

def sayYeh():                     # 매개변수가 없는 함수
    return 'Hi'                   # 매개변수 입력되지 않아도 출력이 된다
                                  # 매개변수, parameter
                                  # 매개변수를 받아들이지 않아도 함수가 만들어진다는 것을 알아두자
aaa = sayYeh()
print(aaa)


def sum2(a, b, c):
    return a + b + c              # return에 반복문, 조건문 등등을 넣어서도 함수를 만들 수 있다!

a = 1
b = 2
c = 34
d = sum2(a, b, c)
print(d)