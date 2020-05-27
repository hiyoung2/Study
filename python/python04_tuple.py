# 2. 튜플, tuple
# 리스트와 거의 같으나, 삽입, 삭제, 수정이 안 된다.
# 고정값을 사용 할 때 쓸 수 있다
# 변경이 안 된다!!!

a = (1, 2 ,3)
b = 1, 2, 3

print(type(a))            # <class 'tuple'>
print(type(b))            # <class 'tuple'>

# a.remove(2)             # 오류 발생! WHY?  
# print(a)                # AttributeError: 'tuple' object has no attribute 'remove'
                          # tuple에는 remove라는 attribute(속성)이 없다

print(a + b)              # 출력 : (1, 2, 3, 1, 2, 3)
print(a * 3)              # 출력 : (1, 2, 3, 1, 2, 3, 1, 2, 3)

# print(a - 3)            # TypeError: unsupported operand type(s) for -: 'tuple' and 'int'

# 튜플은 많이 사용하지 않음, 리스트를 많이 씀
# 튜플은 이 정도로만 알면 된다