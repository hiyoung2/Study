# 2.17 리스트 컴프리헨션

# 기존의 리스트에서 특정 항목을 선택하거나 변환시킨 결과를 새로운 리스트에 지정해야하는 경우도 자주 발생
# 가장 파이썬스럽게 처리하는 방법은 리스트 컴프리헨션이다

even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares      = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]

# 또한 딕셔너리나 집합으로 변환시킬 수 있다

square_dict = {x: x * x for x in range(5)} # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
square_set = {x * x for x in [1, -1]} # {1}

# 보통 리스트에서 불필요한 값은 밑줄로 표기한다
zeros = [0 for _ in even_numbers] # even_numbers와 동일한 길이

# 리스트 컴프리헨션에는 여러 for 함수를 포함할 수 있고

pairs = [(x, y)
          for x in range(10)
          for y in range(10)] # (0,0) (0,1), ... , (9,8), (9,9) 총 100개

# 뒤에 나오는 for는 앞에 나온 결과에 대해 반복한다

increasing_pairs = [(x, y)                      # x < y 인 경우만 해당
                     for x in range(10)         # range(lo, hi)는
                     for y in range(x + 1, 10)] # [lo, lo+1, ..., hi-1]을 의미
