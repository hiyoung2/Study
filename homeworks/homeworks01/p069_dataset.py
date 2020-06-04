# 5장 통계

# 통계는 데이터를 이해하는 바탕이 되는 수리적 기법
# 5. 1 데이터셋 설명하기

# 데이터의 수가 적다면 데이터 자체를 보여주는 것이 가장 좋은 방법일 수있다
# but, 데이터가 많다면? (가령 100만 개의 숫자로 구성) 데이터를 다루는 것도 불편, 이해하기도 힘들 것
# 이럴 때 통계를 사용하면 데이터를 정제해서 중요한 정보만 전달해 줄 수 있다! wow

# 일단 사용자들의친구 수를 Counter와 plt.bar()를 사용해 히스토그램으로 표현해보자

# num_friends = [100, 49, 41, 40, 25, ... # 훨씬 더 많은 데이터]
# 그래프에 이용될 data가 명호가히 제시 되지 않아서 코딩 따라하고 주석 처리

# from collections import Counter
# import matplotlib.pyplot as plt

# friend_counts = Counter(num_friends)
# xs = range(101)                       # 최댓값은 100
# ys = [friend_counts[x] for x in xs]   # 히스토그램의 높이는 해당 친구 수를 갖고 있는 사용자 수
# plt.bar(xs, ys)
# plt.axis([0, 101, 0, 25])             # x 축은 0~101(친구 수 0명에서 100명), y 축은 0~25(사용자수)
# plt.title("Histogram of Friend Counts")
# plt.xlabel("# of friends")
# plt.ylabel('# of people')

# # 히스토그램에 대한 통계치 계산
# # 가장 간단한 통계치는 데이터 포인트의 갯수

# num_points = len(num_friends)   # 204

# # 최댓값과 최솟값도 유용할 것
# largest_value = max(num_friends) # 100 / 친구 가장 많이 가진 사용자의 친구 수?
# smallest_value = min(num_friends) 

# # 한편 최댓값, 최솟값을 구하는 문제는 정렬된 리스트의 특정 위치에 있는 값을 구하는 문제로 볼 수 있다
# sorted_values = sorted(num_friends)
# smallest_value = sorted_values[0]    # 크기 작은 순서대로 가니까 가장 적은 친구의 수는 [0]
# second_smallest_values = sorted_values[1]
# second_largest_values = sorted_values[-2]

