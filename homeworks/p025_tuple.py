# 2.10 튜플

# 튜플, tuple은 변경할 수 없는 리스트이다
# 리스트에서 수정을 제외한 모든 기능을 튜플에 적용시킬 수 있다
# 튜플은 대괄호 대신 괄호()를 사용해서(혹은 아무런 기호 없이) 정의

my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4

print("my_list : ", my_list)
print("my_tuple : ", my_tuple)
print("other_tuple : ", other_tuple)

my_list[1] = 3
print("my_list(new) : ", my_list)

# try:
#     my_tuple[1] = 3
# except TypeError:
#     print("cannot modify a tuple")


# 함수에서 여러 값을 반환할 때 튜플을 사용하면 편하다

def sum_and_product(x, y):
    return (x + y), (x * y)

sp = sum_and_product(2, 3)
print(sp) # (5, 6)
s, p = sum_and_product(5, 10)
print(s) # 15
print(p) # 50

# tuple과 list는 다중 할당(multiple assignment)을 지원한다
x, y = 1, 2
print(x) # 1
print(y) # 2

x, y = y, x
print(x) # 2
print(y) # 1

