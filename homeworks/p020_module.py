# 2.5 모듈
# 파이썬에 기본적으로 포함된 몇몇 기능과 각자 내려받은 제3자 패키지(3rd party packages)에
# 포함된 기능들은 파이썬을 실행시킬 때 함께 실행되지 않는다
# 이 기능을 사용하기 위해서는 모듈을 불러오는 import를 사용해야 한다

import re
my_regx = re.compile("[0-9]+", re.I)

# 여기서 불러온 re는 정규표현식(regular expression, regex)을 다룰 때 필요한
# 다양한 함수와 상수를 포함
# 그 함수와 상수를 사용하기 위해서는 re 다음에 마침표(.)를 붙인 후 함수나 상수의 이름을 이어 쓰면 됨
# 코드에서 이미 re를 사용하고 있다면 별칭(alias)을 사용할 수도 있다

import re as regex
my_regx = regex.compile("[0.9+", regex.I)

# 모듈의 이름이 복잡하거나 이름이 반복적으로 타이핑할 경우에도 별칭을 사용할 수 있다
# 예를 들어 matplotlib라는 라이브러리로 데이터를 시각화 할 때는 다음과 같은 별칭을 관습적으로 사용

import matplotlib
# pltplot(...)

# 모듈 하나에서 몇몇 특정한 기능만 필요하다면 전체 모듈을 불러오지 않고 해당 기능만 명시해서 불러올 수 있다

from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

# 가장 좋지 않은 습관 중 하나는 모듈의 기능들을 통째로 불러와서 기존의 변수들을 덮어쓰는 것이다

# match = 10
# from re import * # re에도 match 라는 함수가 존재 / import * : 모듈 기능 통째로 부르기
# print(match)