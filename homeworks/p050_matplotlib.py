# 3장 데이터 시각화

# 데이터 과학자가 갖춰야 할 기본 기술 중 하나는 데이터 시각화!
# 시각화를 만드는 것은 아주 쉽지만, 좋은 시각화를 만드는 것은 어려움
# 데이터 시각화의 목적?
# - 데이터 탐색(exploration)
# - 데이터 전달(communication)


# 3.1 matplotlib

# 데이터 시각화하기 위한 도구 무궁무진
# 그 중 하나인 matplotlib를 사용해보자
# matplotlib는 웹을 위한 복잡하고 interactive한 시각화를 만들고 싶다면 가장 좋은 선택은 아닐 수도,,
# 간단한 막대 그래프, 선 그래프, 또는 산점도를 그릴 때는 나쁘지 않음
# matplotlib는 파이썬에서 기본으로 제공하는 라이브러리가 아님
# 가상 환경을 활성화하고 아래 명령어로 matplotlib를 설치하자

from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# x축에 연도, y축에 GDP가 있는 선 그래프를 만들자

plt.plot(years, gdp, color ='green', marker='o', linestyle='solid')

# 제목을 더하자
plt.title("Nominal GDP")

# y축에 레이블을 추가하자
plt.ylabel("Billions of $")
plt.show()
