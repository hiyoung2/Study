# 7.3 p-value 

# 가설을 바라보는 또 다른 관점은 p-value이다
# 이는 어떤 확률값을 기준으로 구간을 선택하는 대신에 H0가 참이라고 가정하고
# 실제로 관측된 값보다 더 극단적인 값이 나올 확률을 구하는 것이다

# 동전이 공평한지를 확인하기 위해 양측검정을 해 보자

# 동전 던지기 코드가 안 돌아가서요,,,,,

'''
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1):
    
    # mu(평균)와 sigma(표준편차)를 따르는 정규분포에서 x같이
    # 극단적인 값이 나올 확률은 얼마나 될까?
    
    if x>= mu:
        # 만약 x가 평균보다 크다면 x보다 큰 부분이 꼬리다
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)

# 만약 동전의 앞면이 나온 경우가 530번이 관측되었다면 p-value는 다음과 같다
two_sided_p_value(529.5, mu_0, sigma_0) # 0.062

# 시뮬레이션을 해 보면 우리의 추정값이 그럴듯하다는 것 확인 가능

imprt random

extreme_value_count = 0
for _ in range(1000):
    num_heads - sum(1 if random.random() < 0.5 else 0
                    for _ in range(1000))
    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count += 1

# p-value was 0.062 -> ~62 extreme values out of 1000
assert 59 < extreme_value < 65, f"{extreme_value_count}"

# 계산된 p-value가 5%보다 크기 때문에 귀무가설을 기각하지 않음
# 만약 동전의 앞면이 532번 나왔다면 p-value는 5%보다 작을 것이고 이 경우에 귀무가설을 기가할 것

two_sided_p_value(531.5, mu_0, sigma_0)  # 0.0463

이전 가설검정에 비해 통계를 보는 관점만 다를 뿐 동일한 검정방법
같은 방식으로

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

동전의 앞면이 525번 나왔다면 단측검정을 위한 p-value는 다음과 같이 계산되며

upper_p_value(524.5, mu_0, sigma_0)  # 0.061

귀무가설을 기각하지 않을 것이다
만약 동전의 527번 나왔다면 p-value는 다음과 같이 계산되면 귀무가설을 기각할 것!

upper_p_value(526.5, mu_0, sigma_0) # 0.047



'''