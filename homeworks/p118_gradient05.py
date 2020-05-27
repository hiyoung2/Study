# 8.6 미니배치와 SGD(Stochastic Gradient Descent)

# 데이터 셋이 큰 모델 학습하는 경우 그래디언트 계산 오래 걸림
# 이럴 때는 더 자주 그래디언트만큼 이동하는 방법을 사용하면 됨

# 미니배치 경사 하강법(minibatch graident descent)에서는 전체
# 데이터 셋의 샘플인 미니배치에서 그래디언트를 계산함

from typing import TypeVar, List, Iterator
import random

T = TypeVar('T') # 변수의 타입과 무관한 함수를 생성

def minibatches(dataset: List[T], 
                batch_size: int,
                shuffle: bool = True):
    # dataset에서 batch_sze만큼 데이터 포인트를 샘플링해서 미니배치를 생성
    # 각 미니배치의 시작점인 0, batch_size, 2 * batch_size, ...을 나열
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts) # 미니배치의 순서를 섞는다

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]    