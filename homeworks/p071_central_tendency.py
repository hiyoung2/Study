"""
# 5.1.1 중심 경향성
from typing import List
# 데이터의 중심이 어디 있는지를 나타내는 중심 경향성(central tendency) 지표는 매우 중요!
# 그리고 대부분의 경우 데이터의 값을 데이터 포인트의 갯수로 나눈 평균을 사용하게 된다

# def mean(xs: List[float]):
#     return sum(xs) / len(xs)

# mean(num_friends)    # 7.33333

# 만약 데이터 포인트가 두 개라면 평균은 두 데이터 포인트의 정중앙에 위치한 값일 것이다
# 데이터의 갯수를 추가할수록 평균은 각 데이터 포인트의 값에 따라 이동하게 된다 
# 예를 들어 10개의 데이터 포인트 중 아무 데이터 하나만을 1을 증가시켜도 평균은 0.1이 증가한다
# 가끔은 중앙값(mdedian)도 필요할 것이다
# 데이터 포인트의 갯수가 홀수라면 중앙값은 전체 데이터에서 가장 중앙에 있는 데이터 포인트를 의미
# 반면 짝수라면? 중앙값은 전체 데이터에서 가장 중앙에 있는 두 데이터 포인트의 평균을 의미

# 예를 들어 5개의 데이터 포인트가가 값의 크기에 따라 정렬된 x라는 벡터로 주어졌다!
# 중앙값은 x[5 //2], 즉 x[2]이다
# 만약 6개의 데이터 포인트가 주어졌다면 중앙값은 세 번쩨 데이터 포인트 x[2]와 네 번째 x[3]의 평균이다

# 사실은 평균과 달리 중앙ㄱ밧은 데이터 포인트 모든 값의 영향을 받지 않는다
# 값이 가장 큰 데이터 포인트의 값이 더 커져도 중앙값은 변하지 않는다
# 데이터 포인트의 갯수가 짝수인 경우를 포함해야 하기 때문에 median 함수는 조금 복잡하다

# 밑줄 표시로 시작하는 함수는 프라이빗private 함수를 의미
# median 함수를 사용하는 사람이 직접 호출하는 것이 아닌
# median 함수만 호출하도록 생성됨

def _median_odd(xs: List[float]):
    # len(xs) 가 홀수이면 중앙값을 반환
    return sorted(xs[len(xs)] // 2)

def median_even(xs: List[float]):
    # len(xs)가 짝수이면 두 중앙값의 평균을 반환
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2 # e.g. length 4 >> hi_midpoint 2
    return (sorted_xs[hi_midpoint -1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]):
    # v의 중앙값을 계산
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2

# 이제 사용자별 친구 수의 중앙값을 계산해 볼 수 있다
print(median(num_friends))

# 평균은 중앙값보다 게산하기 간편하며 데이터가 바뀌어도 값의 변화가 더 부드럽다
# 만약 n개의 데이터 포인트가 주어졌을 때 데이터 포인트 한 개의 값이 작은 수 e만큼 증가한다면
# 평균은 e/n만큼 증가할 것이다
# (이러한 성질 덕분에 평균에 다양한 미적분 기법을 적용할 수 있다)
# 반면 중앙값을 찾기 위해선 주어진 데이터를 정렬해야 함
# 만약 데이터 포인트 한 개의 값이 작은 수 e 만큼 증가한다면 중앙값은 e만큼 증가할 수도 있고
# e보다 작은 값만큼 증가할 수도 있다
# 심지어 주어진 데이터에 따라 중앙값이 변하지 않을 수도 있다

# 하지만 평균은 이상치, outlier에 매우 민감하다
# 가령, 네트워크에서 친구가 가장 많은 사용자가 200명이라 치자
# 이런 경우 평균은 7.82 만큼 증가 but 중앙값은 변하지 않음
# 이상치가 '나쁜' 데이터(이해하려는 현상을 제대로 나타내고 있지 않은 데이터)라면
# 평균은 데이터에 대한 잘못된 정보를 줄 수도 ,,,,
# 1980년대 노스캐롤라이나 대학교의 전공 중 지리학과 졸업생이 초봉 가장 높게 나옴
# 그 이유는 지리학을 전공한 NBA 마이클 조던의 초봉 때문,,, ㅋㅋㅋㅋㅋ 
# 또 분위(quantile)는 중앙값을 포괄하는 개념인데 특정 백분위보다 낮은 분위에 속하는 데이터를 의미
# 중앙값은 상위 50%의 데이터보다 작은 값을 의미

def quantile(xs: List[float], p: float):
    # x의 p분위에 속하는 값을 반환
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13

# 흔치는 않지만 최빈값(mode, 데이터에서 자주 나오는 값)을 살펴보는 경우도 있다
def mode(x: List[float]) :
    # 최빈값이 하나보다 많을 수도 있으니 결과는 리스트로 반환
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == ma_count]

assert set(mode(num_friends)) == {1, 6}

"""
