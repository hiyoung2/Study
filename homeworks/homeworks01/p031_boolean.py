# 2.15 True와 False

# 다른 프로그래밍 언어처럼 파이썬에도 불(boolean) 타입이 존재하는데
# 이들은 항상 대문자로 인식한다

aaa = one_is_less_than_two = 1 < 2
print(aaa) # True

bbb = true_equals_false = True == False
print(bbb) # False

# 다른 언어의 null처럼 파이썬은 존재하지 않는 값을 None이라고 표기한다

x = None
assert x == None, "this is the not the Pythonic way to check for None"
assert x is None, "this is the Pythonic way to check for None"

# 파이썬은 다른 값으로도 불 타입을 표현할 수 있게 해준다
# 다음은 모두 거짓을 의미한다
############ * False
############ * None
############ * [](빈 리스트)
############ * {}(빈 딕셔너리)
############ * ""
############ * set()
############ * 0
############ * 0.0

# 나머지 거의 모든 것은 참, True을 의미한다
# 이를 통해 리스트, 문자열, 딕셔너리 등이 비어 있는지 쉽게 확인할 수 있다
# 하지만 예상치 못한 오류가 발생하기도 한다

# s = some_function_that_returns_a_string()
# if s:
#     first_char = s[0]
# else :
#     first_char = ""

# 위의 코드는 다음과 같이 더욱 간단하게 표현 할 수 있다
# first_char = s and s[0]

# # and는 첫 번째 값이 참이면 두 번째 값을 반환해주고
# # 첫 번째 값이 거짓이면 첫번째 값을 반환해준다
# # 만약 x가 숫자거나 None이라면
# safe_x = x or 0
# # 위의 값은 항상 숫자일 것이다, 하지만
# safe_x = x if x is not None else 0
# 이렇게 표현하는 것이 더 읽기 편할 것이다

# 파이썬에는 리스트의 모든 항목이 참이라면 True를 반환해 주는 all 함수와
# 적어도 하나의 항목이 참이라면 True를 반환해주는 any함수가 있다

all([True, 1, {3}]) # True
all([True, 1, {}])  # False, {}는 False이니까
any([True, 1, {}])  # True
all([]) # True, 거짓인 항목이 없기에
any([]) # False, 참인 항목이 없기에

