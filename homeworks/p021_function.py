# 2. 6 함수

# 함수란 0개 혹은 그 이상의 인자를 입력 받아 결과를 반환하는 규ㅣㄱ
# 파이썬에서는 def 를 이용해 함수를 정의

def double(x):
    # 이 곳은 함수에 대한 설명을 적어 놓는 공간
    # 예를 들어, '이 함수는 입력된 변수에 2를 곱한 값을 출력해 준다'라는 설명을 추가 할 수 있다
    return x * 2 

# 파이썬 함수들은 변수로 할당되거나 함수의 인자로 전달할 수 있다는 점에서 일급 함수(first-class)의 특성을 가짐

def apply_to_one(f):
    # 인자가 1인 함수 f를 호출
    return f(1)

my_double = double # 방금 정의한 함수를 나타낸다
x = apply_to_one(my_double)
print(x) # 2

# 짧은 익명의 람다 함수(lambda function)도 간편하게 만들 수 있다

y = apply_to_one(lambda x: x + 4)
print(y) # 5

# 대부분의 사람들은 그냥 def를 사용하라고 이야기하겠지만 변수에 람다 함수를 할당할 수도 있다

# another_double = lambda x: 2 * x  <- 이 방법은 최대한 피하도록 하자

def another_double(x):
    return 2 * x       # 대신 이렇게 작성하자

# 함수의 인자에는 기본값을 할당할 수 있는데 기본값 외의 값을 전달하고 싶을 때는 값을 직접 명시해주면 된다

def my_print(message = "my default message"):
    print(message)

my_print("hello") # hello 출력
my_print()        # my default message 출력

# 가끔씩 인자의 이름을 명시해 주면 편리하다
def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last

full_name("Joel", "Grus") # Joel Grus 출력
full_name("Joel")         # Joel Something 출력  
full_name(last = "Grus")  # What's-his-name Grus 출력 

