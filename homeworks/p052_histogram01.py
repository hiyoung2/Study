# 3.2 막대그래프

# 막대 그래프, bat chart는 이산적인(discrete) 항목들에 대한 변화를 보여줄 때 사용하면 좋다
# 예를 들어 여러 영화가 아카데미 시상식에서 상을 각각 몇 개씩 받았는지 보여 준다

from matplotlib import pyplot as plt

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 [0, 1, 2, 3, 4], y 좌표는 [num_oscars]로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")    # 제목을 추가
plt.ylabel("# of Academy Awards")  # y축에 레이블을 추가

# x축 각 막대의 중앙에 영화 제목을 레이블로 추가
plt.xticks(range(len(movies)), movies)

plt.show()