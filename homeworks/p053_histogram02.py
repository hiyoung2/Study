# 막대 그래프를 이용하면 히스토그램, histogram도 그릴 수있다
# 히스토그램이란 정해진 구간에 해당하는 항목의 갯수를 보여줌으로써
# 값의 분포를 관찰할 수 있는 그래프 형태이다


from matplotlib import pyplot as plt
from collections import Counter
grades = [83, 95, 91, 87, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화하고 100점은 90점대에 속한다
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # 각 막대를 오른쪽으로 5만큼 옮기고
         histogram.values(),                # 각 막대의 높이를 정해 주고
         10,                                # 너비는 10으로 하자
         edgecolor=(0, 0, 0))               # 각 막대의 테두리는 검은색으로 설정

plt.axis([-5, 105, 0, 5])   # x축은 -5부터 106
                            # y축은 0부터 5

plt.xticks([10 * i for i in range(11)]) # x축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
# 막대 그래프를 이용하면 히스토그램, histogram도 그릴 수있다
# 히스토그램이란 정해진 구간에 해당하는 항목의 갯수를 보여줌으로써
# 값의 분포를 관찰할 수 있는 그래프 형태이다


from matplotlib import pyplot as plt
from collections import Counter
grades = [83, 95, 91, 87, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화하고 100점은 90점대에 속한다
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # 각 막대를 오른쪽으로 5만큼 옮기고
         histogram.values(),                # 각 막대의 높이를 정해 주고
         10,                                # 너비는 10으로 하자
         edgecolor=(0, 0, 0))               # 각 막대의 테두리는 검은색으로 설정

plt.axis([-5, 105, 0, 5])   # x축은 -5부터 106
                            # y축은 0부터 5

plt.xticks([10 * i for i in range(11)]) # x축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()


# plt.bar의 세 번째 인자(argument)는 막대의 너비를 정한다
# 여기서는 각 구간의 너비가 10이므로 막대의 너비 또한 10으로 설정
# 또 막대들을 오른쪽으로 5씩 이동해서 
# (예를 들어) '10'에 해당하는 막대의 중점이 15가 되게 했다
# 막대 간 구분이 되도록 각 막대의 테두리를 검은색으로 설정

# plt.axis는 x축의 범위를 -5에서 105로 하고 ('0', '100'에 해당하는 막대가 잘리지 않도록 하기 위해)
# y축의 범위를 0부터 5로 정했다
# 그리고 plt.xticks는 x축의 레이블이 0, 10, 20, ..., 100이 되게 했다
# plt.axis를 사용할 때에는 특히 신중해야 함
# 막대 그래프를 그릴 때 y축을 0에서 시작하지 않으면
# 아래 코딩의 결과 그래프와 같이 오해를 불러일으키기 쉽기 때문
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# 이렇게 하지 않으면 matplotlib이 x축에 0, 1 레이블을 달고
# 주변부 어딘가에 2+013e3이라고 표기해 둘 것
plt.ticklabel_format(useOffset=False)

# 오해를 불러일으키는 y축은 500 이상의 부분만 보여 줄 것이다
# plt.axis([2016.5, 2018.5, 499, 506])
# plt.title("Look at the 'Huge' Increase!")
# plt.show()

# 위의 그래프보다 더 적합한 축을 사용해서 합리적인 그래프를 만들자
plt.axis([2016.5, 2018.5, 0, 550]) # y 축을 0에서 시작함을 설정해둠!
plt.title("Not So Huge Anymore")
plt.show()