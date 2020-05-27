# 7.5 p해킹

# 귀무가설을 잘못 기각하는 경우 5%인 가설검정은 정의에서 알 수 있듯이 모든 경우의 5%에서 귀무가설을 잘못 기각한다
'''
from typing import List
import random

def run_experiment():
    # 동전을 1000번 던져서 True = 앞면, False = 뒷면
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]):
    # 신뢰구간을 5%로 설정
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_refections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])


assert num_rejections == 46

'''