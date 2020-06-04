# 2.24 zip과 인자 언패킹

# 가끔씩 두 개 이상의 리스트를 서로 묶어주고 싶을 때가 있다
# zip은 여러 개의 리스트를 서로 상응하는 항목의 튜플로 구성된 리스크로 변환해준다

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# 실제 반복문이 시작되기 전까지는 묶어주지 않는다
[pair for pair in zip (list1, list2)]   # [('a', 1), ('b', 2), ('c', 3)]

# 주어진 리스트의 길이가 서로 다른 경우 zip은 첫번째 리스트가 끝나면 멈춘다
# 묶인 리스트는 다음과 같은 트릭을 사용해 다시 풀어줄 수도 있다

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

# 여기서 사용한 * 별표는 원래 인자 언패킹(argument unpacking)을 할 때 사용되는 문법
# 이를 사용하면 pairs 안의 항목들을 zip 함수에 개별적인 인자로 전달해준다
# 아래 코드와 같은 것이다
letters, numbers = zip(('a', 1), ('b', 2), ('c', 3))

# 이런 방식의 인자 해체는 모든 함수에 적용할 수 있다
def add(a, b): return a + b

# add(1, 2)      # 3 
# try:
#     add([1, 2])
# except TypeError:
#     print("add expects two inputs")
# add(*[1, 2])   # 3
