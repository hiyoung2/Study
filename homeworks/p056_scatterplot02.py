# 변수들끼리 비교할 대 matplotlib이 자동으로 축의 범위를 설정하게 하면
# 공정한 비교를 하지 못하게 될 수도 있다

from matplotlib import pyplot as plt

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")

plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")

# plt.axis("equal")
plt.show()



# 여기서 plt.axis("equal")이라는 명령을 추가하면 공ㅈ어한 비교를 할 수 있게 된다
