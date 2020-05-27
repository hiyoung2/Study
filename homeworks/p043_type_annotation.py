# 2.26 타입 어노테이션
# annotation : 주석

# 파이썬은 동적 타입, dynamically typed 언어다
# 이는 변수만 올바르게만 사용한다면 변수의 타입은 신경 쓰지 않아도 된다는 뜻

'''
def add (a, b) :
    return a + b

assert add(10, 5) == 15,                   "+ is valid for numbers"
assert add([1, 2], [3]) == [1, 2, 3],      "+ is valid for lists"
assert add("hi", "there") == "hi there",   "+ is valid for strings"

try :
    add(10, "five")
except TypeError:
    print("cannot add an int to a string")


# 2.26 타입 어노테이션 하는 방법
# int, bool, float 같은 기본적인 객체는 타입을 바로 명시해 주면 된다
# 리스트의 경우에는 어떻게 타입을 명시하는 게 좋을까?

def total(xs: list): # ->float:
    return sum(total)

# xs는 문자열이 아닌 float 객체를 가지고 있는 리스트
# typiing 모듈을 사용하면 이렇게 구체적으로 타이을 명시할 수 있다
from typing import List
def total(xs: List[float]):
    return sum(total)


# 지금까지는 변수의 타입이 너무 명확했기 때문에 함수의 인자나 반환값에 대해서만 타입을 명시했다

# 이렇게 변수의 탕비을 명시할 수 있다
# 하지만 x가 int라는 것이 너무 명확하기 때문에 함수의 인자나 반환값에 대해서만 타입을 명시했다

x : int = 5

# 종종 변수의 타입이 명확하지 않을 때가 있다
values = []
best_so_far = None
# 이 변수들은 무슨 타입?
# 이런 경우에는 변수를 정의할 때 타입에 대한 힌틀를 추가할 수가 있다

from typing import Optional
values: List[int] = []
best_so_far: Optional[float] = None # float이나 None으로 타입 명시

# typing 모듈은 다양한 타입을 제공하지만 그 중에서 몇 개만 자주 사용
# 여기서 명시하고 있는 타입들은 너무 자명하여 굳이 명시할 필요가 없다
from typing import Dict, Iterable, Tuple

# 키는 문자열, 값은 int
counts: Dict[str, int] = {'data':1, 'science':2}

# 리스트와 제너레이터는 모두 이터러블이다
lazy = []
if lazy :
    evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
else :
    evens = [0, 2, 4, 6, 8]

# 튜플 안의 각 항목들의 타입을 구체적으로 명시
triple: Tuple[int, float, int] = (10, 2.3, 5)

# 파이썬의 일급 함수(first-class fucntions)에 대해서도 타입을 명시할 수 있다
# 인위적으로 만든 예시

from typing import Callablle

# repeater 함수가 문자열과 int를 인자로 받고
# 문자열을 반환해 준다는 것을 명시
def twice(repeater: Callable[[str, int], str], s: str): # -> str:
    return repeater(s, 2)

def comma_repeater(s: str, n: int): # -> str:
    n_copies = [s for _ in range(n)]
    return ', '.join(n_copies)

assert twice(comma_repeater, "type hints") == "type hints, type hints"

# 명시된 타입 자체도 파이썬 객체이기 때문에 벼수로 선언할 수 있다

Number = int
Numbers = List[Number]

def total(xs: Numbers): # ->Number:
    return sum(xs)

'''