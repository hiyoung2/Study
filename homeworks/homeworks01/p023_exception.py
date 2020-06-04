# 2.8 예외 처리
# 코드가 뭔가 잘못됐을 때 파이썬은 예외(exception)가 발생했음을 알려줌
# 예외를 제대로 처리해주지 않으면 프로그램이 죽는데, 이를 방지하기 위해 사용할 수 있는 것이
# try와 except이다

try : 
    print(0 /0)
except ZeroDivisionError:
    print("cannot divide by zero")

# 많은 프로그래밍 언어에서 예외는 나쁜 것이라고 받아들여지지만
# 파이썬에서는 코드를 깔끔하게 작성하기 위해서라면 얼마든지 사용된다