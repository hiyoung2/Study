# 2.20 이터레이터와 제너레이터

# 리스트는 순서나 인덱스만 알고 있으면 쉽게 특정 항목을 가져올 수 있다는 큰 장점이 있음
# 하지만 이러한 장점은 경우에 따라 큰 단점이 될 수도,,,
# 가령 10억 개의 항목으로 이루어진 리스트를 생성하려면 컴퓨터의 메모리가 부족해 질 수 있다
# 만약 항목을 하나씩 처리하고 싶다면 리스트 전체를 가지고 있을 필요가 없다
# 앞부분의 몇몇 값만 필요한데도 10억 개의 항목을 갖는 리스트 전체를 생성하는 것은 매우 비효율적이다

# 제너레이터, generator는 (주로 for문을 통해서) 반복할 수 있으며, 제너레이터를 만드는 한 가지 방법은
# 함수와 yield를 사용하는 것이다

def generate_range(n):
    i = 0
    while i < n:
        yield i # yield가 호출될 때마다 제너레이터에 해당 값을 생성
        i += 1

# 다음과 같은 반복문은 yield로 반환되는 값이 없을 때까지 반환된 값을 차례로 하나씩 사용한다
for i in generate_range(10):
    print(f"i: {i}")

# (사실, range 자체가 제너레이터로 만들어졌기 때문에 이렇게 따로 만들 필요는 없다,,,,예?)


# 이는 무한한 수열도 메모리의 제약을 받지 않고 구현할 수 있다는 것을 의미한다
def natural_numbers():
    # 1, 2, 3, .......... 을 반환
    n = 1
    while True:
        yield n
        n += 1
# 물론 break 없이 무한 수열을 생성하는 것은 추천하지 않는다

# 제너레이터의 단점은 제너레이터를 단 한 번만 반복할 수 있다는 점!
# 만약 데이터를 여러 번 반복해야 한다면, 제너레이터를 다시 만들거나 리스트를 사용해야 한다
# 제너레이터를 매번 생성하는 것이 너무 오래 걸린다면 리스트 사용을 추천!

# 또한 괄호 안에 for문을 추가하는 방법으로도 generator를 만들 수 있다
evens_below_20 = (i for i in generate_range(20) if i % 2 == 0)

# 물론 for나 next를 통해서 반복문이 시작되기 전까지는 제너레이터가 생성되지 않는다
# 이를 사용하여 정교한 데이터 처리 파이프라인을 만들 수 있다
 
 # 실제 반복문이 시작되기 전까지는 제너레이터가 생성되지 않는다
data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)
# 등등

# 종종 리스트나 제너레이터에서 항목을 하나씩 확인해 볼 경우 항목의 순서(index)를 반환하고 싶을 때도 있다
# 파이썬의 enumerate 함수를 사용하면 (순서, 항목) 형태로 값을 반환시킬 수 있다

names = ["Alice", "Bob","Charlie", "Debbie"]

# 파이썬스럽지 않다(파이썬스러운 , 파이썬스럽다,,,, 되게 좋아하네)
for i in range(len(names)):
    print(f"name {i} is {names[i]}")

# 파이썬스럽지 않다
i = 0
for name in names:
    print(f"name {i} is {name[i]}")

# 파이썬스럽다(드디어)
for i, name in enumerate(names):
    print(f"name {i} is {name}")