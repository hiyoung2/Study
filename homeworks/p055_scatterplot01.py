# 3.4 산점도

# 산점도(scatterplot)은 두 변수 간의 연관 관계를 보여 주고 싶을 때 적합한 그래프
# 예를 들어 각 사용자의 친구 수와 그들이 매일 사이트에서 체류하는 시간 사이의 연관성을 보여주는 그래프 같은 것

from matplotlib import pyplot as plt

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# 각 포인트에 레이블을 달자
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy = (friend_count, minute_count),
        xytext=(5, -5),
        textcoords = 'offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()


