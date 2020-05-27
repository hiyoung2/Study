# 2.22 정규표현식

# 정규표현식(regular expressions, regx)을 사용하면 문자열을 찾을 수 있다
# 굉장히 복잡한 것
# 간략한 예시만 보자

import re

re_examples = [                               # 모두 True
    not re.match("a", "cat"),                 # 'cat'은 'a'로 시작하지 않기 때문에
    re.search("a", "cat"),                    # 'cat; 안에는 'a'가 존재하기 때문에
    not re.search("c", "dog"),                # 'dog' 안에는 'c'가 존재하지 않기 때문에
    3 == len(re.split("[ab]", "carbs")),      # a 혹은 b 기준으로 분리하면
                                              # ['c', 'r', 's']가 생성되기 때문에
    "R-D-" == re.sub("[0-9]", "-", "R2D2")    # 숫자를 "-"로 대체하기 때문에
]

assert all(re_examples), "all the regx examples should be True"

# re.match 메서드는 문자열의 시작이 정규표현식과 같은지 비교하고 
# re.search 메서드는 문자열 전체에서 정규표현식과 같은 부분이 있는지 찾는다
