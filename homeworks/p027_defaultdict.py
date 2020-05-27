# 2.11.1 decaultdict

# 문자에서 단어의 빈도수를 세어 보는 중이라고 생각해보자
# 가장 직관적인 방법은 단어를 키로, 빈도수를 값으로 지정하는 딕셔너리를 생성하는 것
# 이 때, 각 단어가 딕셔너리에 이미 존재하면 값을 증가시키고
# 존재하지 않는다면 새로운 키와 값을 추가해주면 된다

document = ["word", "world", "word", "password"] # 임의로 내가 설정

word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

print(word_counts) # {'word': 2, 'world': 1, 'password': 1}

# 혹은 '용서를 구하는 게 허락을 받는 것보다 쉽다(forgiveness is better than permission)'는 마음가짐으로
# 예외를 처리하면서 딕셔너리를 생성하는 방법도 있다
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

# 세 번째 방법은 존재하지 않는 key를 적절하게 처리해 주는 get을 사용해서 딕셔너리를 생성하는 방법

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

# 세 가지 방법 모두 약간 복잡
# 이런 경우 defaultdict를 사용하면 편해진다
# defaultdict와 평범한 딕셔너리의 유일한 차이점은 만약 존재하지 않는 키가 주어진다면
# defaultdict는 이 키와 인자에서 주어진 값으로 dict에 새로운 항목을 추가해 준다는 것이다
# defaultdict를 사용하기 위해서는 먼저 collections 모듈에서 defaultdict를 불러와야 한다

from collections import defaultdict

word_counts = defaultdict(int) # int()는 0을 생성
for word in document:
    word_counts[word] += 1

# 리스트, 딕셔너리 혹은 직접 만든 함수를 인자에 넣어줄 수 있다

dd_list = defaultdict(list) # list()는 빈 리스트를 생성
dd_list[2].append(1) # 이제 dd_list는 {2: [1]}을 포함
print(dd_list) # defaultdict(<class 'list'>, {2: [1]})

dd_dict = defaultdict(dict)
dd_dict["Joel"]["City"] = "Seattle" 
print(dd_dict) # defaultdict(<class 'dict'>, {'Joel': {'City': 'Seattle'}})

# 만약 키를 사용해서 어떤 결과를 수집하는 중이라면 
# 매번 키가 존재하는지 확인할 필요없이 딕셔너리를 생성할 수 있다

