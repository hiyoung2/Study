# 2.21 난수 생성

# 난수 생성시 random 모듈 사용

import random
random.seed(10) # 매번 동일한 결과를 반환해주는 설정
four_uniform_randoms = [random.random() for _ in range(4)]
print(four_uniform_randoms)

# 출력 : [0.5714025946899135, 0.4288890546751146, 0.5780913011344704, 0.20609823213950174]

# random.random()은 0과 1 사이의 난수를 생성
# 앞으로 자주 사용할 함수

# 만약 수도 랜덤(pseudo random)한(결정론적으로 동일한) 난수를 계속 사용하고 싶다면
# random.seed를 통해 매번 고정된 난수를 생성하면 된다

random.seed(10)         # seed를 10으로 설정
print(random.random())  # 0.5714025946899135 
random.seed(10)
print(random.random())  # 0.5714025946899135 동일하게 출력

# 인자가 1개 혹은 2개인 random.randrange 메서드를 사용하면 
# range()에 해당하는 구간 안에서 난수를 생성할 수 있다

random.randrange(10)     # range(10) = [0, 1, ...., 9]에서 난수 생성
random.randrange(3, 6)   # range(3, 6) = [3, 4, 5]

# random 모듈에는 가끔씩 사용하지만 유용한 여러 함수가 존재
# random.shuffle은 리스트의 항목을 임의의 순서로 재정렬해 준다

up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)
# 출력 : [5, 6, 9, 2, 3, 7, 8, 4, 1, 10] (사람마다 결과가 다르게 나옴)

# random.choice 메서드를 사용하면 리스트에서 임의의 항목을 하나 선택할 수 있다

my_best_friend = random.choice(["Alice", "Bob", "Charlie"]) 
print(my_best_friend) # 출력 : Bob

# rnadom.sample을 사용하면 리스트에서 중복이 허용되지 않는 임으의 표본 리스트를 만들 수 있다

lottery_numbers = range(60) # 복권 번호를 무작위로 뽑을 수 있겠네
winning_numbers = random.sample(lottery_numbers, 6)
print(winning_numbers) # 출력 : [4, 15, 47, 23, 2, 26]

# 만약 중복이 허용되는 임의의 표본 리스트를 만들고 싶다면 random.choice 메서드를 여러 번 사용하면 된다
four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)  # 출력 : [6, 6, 4, 0]