# 5.2 상관관계

# "사용자가 사이트에서 보내는 시간과 사용자의 친구 수 사이에 연관성이 있다"라는 가설 검증 요청
# 사이트 사용량 데이터를 통해 각 사용자가 하루에 몇 분 동안 데이텀을 사용하는지 : daily_minutes 리스트 작성
# 이 리스트의 각 항목과 num_friends 리스트의 각 항목이 같은 사용자를 의미하도록 리스트 정렬함
# 두 리스트의 관계는??

# 일단, 분산과 비슷한 개념인 공분산, covariation 살펴보기
# 분산은 하나의 변수가 평균에서 얼마나 멀리 떨어져 있는지 계산한다면
# 공분산은 두 변수가 각각의 평균에서 얼마나 멀리 떨어져 있는지 살펴본다
# num_friends = 
'''
from typing import List

from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]):
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs)-1)

assert 22.42 < covariance(num_friends, daily_minuts) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 /60

# dot은 각 성분별로 곱한 값을 더해 준다는 것을 기억
# 만약 x와 y 모두 각각의 평균보다 크거나 작은 경우, 양수가 더해질 것
# 반면 둘 중 하나는 평균보다 크고 다른 하나는 평균보다 작을 경우, 음수가 더해질 것
# 공분산이 양수이면 x의 값이 클수록 y의 값이 크고, x의 값이 작을수록 y의 값도 작다
# 공분산이 음수이면 x의 값이 클수록 y의 값이 작고, x의 값이 작을수록 y의 값이 크다는 것을 의미
# 공분산이 0이면 그와 같은 관계가 존재하지 않음을 의미

# 그러나,,,,,,,,,,
# 공분산을 해석하는 것은 이러한 이유들로 not easy
# 공분산의 단위는 입력 변수의 단위들을 곱해서 계산되기 때문에 이해하기 쉽지 않음
# - 친구 수와 하루 사용량(분)을 곱한 단위는 무엇을 의미???
# 만약 모든 사용자의 하루 사용량은 변하지 않고 친구 수만 두 배로 증가한다면
# 공분산 또한 두 배로 증가, 하지만 생각해 보면 두 변수의 관계는 크게 변하지 않았음
# 다르게 얘기하면 공분산의 절대적인 값만으로는 '크다'고 판단하기 어려움

# 이러한 이유들로 공분산에서 각각의 표준편차를 나눠 준 상관관계(correlation)을 더 자주 살펴봄

def correlation(xs: List[float], ys: List[float]):
    # xs와 ys의 값이 각각의 평균에서 얼마나 멀리 떨어져 있는지 계산
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0

assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
assert 0.24 < correlation(num_friends, daily_hours) < 0.25

# 상관관계는 단위가 없으며 항상 -1(완벽한 음의 상관관계)에서 1(완벽한 양의 상관관계) 사이의 값을 가짐
# 예를 들어 상관관계가 0.25라면 상대적으로 약한 상관관계를 의미

# 100명의 친구가 있지만 하루에 1분만 사이트를 이용하는 사용자 : 확실한 이상치
# 상관관계에 큰 영향을 미친
# 이 사용자를 제외한다면?


num_friends_good = [x 
                    for i, x in enumerate(num_friends)
                    if i != outlier]
daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58

# 이상치를 제거하면 더 강력한 상관관계를 볼 수 있다
# 이상치 데이터는 알고 보니 회사에서 테스트용으로 생성했다가 제거하는 것을 잊은 내부 테스트용 계정이었다
# 상관관계에서 이상치 사용자를 포함하지 않는 것이 타당하다고 확인되는 순간이었다,,,,,,예? ㅋㅋㅋㅋㅋㅋㅋㅋㅋ

'''