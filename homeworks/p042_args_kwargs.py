# 2.25 args와 kwargs
'''
# 특정 함수 f를 입력하면 f의 결과를 두 배로 만드는 함수를 반환해주는 고차 함수를 만들고 싶다?
def doubler(f): 
    # f를 참조하는 새로운 함수
    def g(x):
        return 2 *f(x)
    # 새로운 함수를 반환
    return g

# 이 함수는 특별한 경우에만 작동한다
def f1(x):
    return x + 1

g = doubler(f1)
assert g(3) == 8, "(3 + 1) * 2 should equal 8"
assert g(-1) == 0, "(-1 + 1) * 2 should equal 0"

# 두 개 이상의 인자를 받는 함수의 경우에는 문제가 발생한다
def f2(x, y):
    return x + y

g = doubler(f2)
try:
    g(1, 2)
except TypeError:
    print("as defined, g only takes one argument")

# 문제를 해결하기 위해 임의의 수의 인자를 받는 함수를 만들어줘야 한다
# 앞서 설명한 인자 어패킹을 사용하면 마법같이(그래서 변수명  magic?)임의의 수의 인자를 받는 함수를 만들 수 있디

def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

# 다음과 같은 결과가 출력
# unnamed args: (1, 2)
# keyword args: {'key': 'word', 'key2': 'word2'}

# 위의 함수에서 args는 이름이 없는 인자로 구성된 튜플이며
# kwargs는 이름이 주어진 인자로 구성된 딕셔너리이다
# 반대로 정해진 수의 인자가 있는 함수를 호출할 대도 리스트나 딕셔너리로 인자를 전달할 수 있다

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 should be 6"

def doulber_correct(f):
    # f의 인자에 상관없이 작동한다
    def g(*args, **kwargs):
        # g의 인자가 무엇이든 간에 f로 보내준다
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6,"doubler should work now"

# 코드의 가독성을 위해 함수에서 필요한 인자는 모두 명시하는 것을 추천
'''