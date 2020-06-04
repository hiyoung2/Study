# 2.13 Set

# 집합(set)은 파이썬의 데이터 구조 중 유일한 항목의 집합을 나타내는 구조이다
# 집합은 중괄호를 사용해서 정의한다

primes_below_10 = {2, 3, 5, 7}
# {}는 비어 있는 딕셔너리를 의미하기 때문에 set()을 사용해서 비어 있는 set을 생성할 수 있다

s = set()
s.add(1)
print(s) # {1}
s.add(2)
print(s) # {1, 2}
s.add(2)
print(s) # {1, 2}

x = len(s)
y = 2 in s
z = 3 in s

print(x) # 2
print(y) # True
print(z) # False

# 두 가지 장점 때문에 앞으로 가끔 집합을 사용할 것
# 첫 번째로 in은 집합에서 굉장히 빠르게 작동한다
# 수많은 항목 중에서 특정 항목의 존재 여부를 확인해 보기 위해서는
# 리스트를 사용하는 것보다 집합을 사용하는 것이 훨씬 효율적

stopwords_list = ["a", "an", "at"] # + hundreds_of_other_words + ["yet", "you"]

"zip" in stopwords_list # False, 하지만 모든 항목을 확인해야 한다

stopwords_set = set(stopwords_list)
"zip" in stopwords_set

# 두 번째 이유는 중복된 원소를 제거해 주기 때문이다

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)
item_set = set(item_list)
print(num_items) # 6
print(item_set) # {1,2, 3}

num_distinct_items = len(item_set)
distinct_item_list = list(item_set)
print(num_distinct_items) # 3
print(distinct_item_list) # {1, 2, 3}


