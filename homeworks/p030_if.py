# 2.14 흐름제어

# 대부분의 프로그래밍 언어처럼 if를 사용하면 조건에 따라 코드를 제어할 수 있다

if 1 > 2:
    message = "if only 1  were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you wnat to)"

# 앞으로도 가끔씩 사용하겠지만 삼항 연산자(ternary operator)인 if-then-else문을 한 줄로 표현 할 수있다
x = 100
parity = "even" if x % 2 == 0 else "odd"

# 파이썬에도 while이 존재하지만

x = 0
while x < 10 :
    print(f"{x} is less than 10")
    x += 1

##출력결과##
# 0 is less than 10
# 1 is less than 10
# 2 is less than 10
# 3 is less than 10
# 4 is less than 10
# 5 is less than 10
# 6 is less than 10
# 7 is less than 10
# 8 is less than 10
# 9 is less than 10

# 다음과 같이 for와 in을 더 자주 사용할 것이다

# range(10)은 0붙 9까지를 의미한다

for x in range(10):
    print(f"{x} is less than 10")

##출력결과##
# 0 is less than 10
# 1 is less than 10
# 2 is less than 10
# 3 is less than 10
# 4 is less than 10
# 5 is less than 10
# 6 is less than 10
# 7 is less than 10
# 8 is less than 10
# 9 is less than 10    

# 만약 더 복잡한 논리 체계가 필요하다면 continue와 break를 사용할 수 있다

for x in range(10):
    if x == 3:
        continue # 다음 경우로 넘어간다
    if x == 5:
        break # for문 전체를 끝낸다
    print(x)

##출력결과##
# 0
# 1
# 2
# 4

