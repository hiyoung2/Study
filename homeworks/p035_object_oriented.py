# 2.19 객체 지향 프로그래밍

# 다른 프로그래밍 언어처럼 파이썬에도 클래스(class)를 사용해서
# 객체 지향 프로그래밍(object-oriented programming)을 하면
# 데이터와 관련 함수를 하나로 묶어줄 수 있다
# 코드를 더 깔끔하고 간단하게 작성하기 위해 클래스를 가끔씩 사용할 예정(책에서)
# 예제를 만들어보자
# '데이터 과학 고급 강의' 같은 모임에 몇 명이 참가했는지 확인해주는
# CountingClicker 클래스를 만들 것이다

# 이 클래스에는 참석자 수를 나타내는 count 변수, count를 증가시키는 click 메서드,
# 현재 count를 반환해주는 read 메서드 그리고 count를 0으로 재설정해주는 reset 메서드가 필요할 것이다
# (실제로 참석자를 셀 때 9999 다음에 0000으로 넘어가는 카운팅 기계를 사용하기도 하지만 이런 경우는 무시히겠음)

# 먼저 클래스를 정의하기 위해서는 class 뒤에 파스칼케이스(PascalCase)로 클래스 이름을 표기하면 된다
# 파스칼케이스란? 공백 없이 여러 단어를 붙여서 표현할 때 각 단어의 첫 글자를 대문자로 표현하는 방식!
'''
class CountingClicker:
    # 함수처럼 클래스에도 주석을 추가할 수 있다!

# 클래스는 0개 이상의 멤버 함수를 포함한다
# 모든 멤버 함수의 첫 번째 인자는 해당 클래스의 인스턴스(instance)를 의미하는 self로 정의해야 한다

def __init__(self, count = 0):
    self.count = count

# 클래스의 이름으로 클래스의 인스턴스를 생성할 수 있다
clicker1 = CountingClicker()           # count = 0 으로 생성된 인스턴스
clikcer2 = CountingClicekr(100)        # count = 100 으로 생성된 이스턴스
clicker3 = CountingClicker(count=100)  # 동일하지만 더욱 명시적인 표현

# __init__메서드 이름의 앞뒤로 밑줄 표시(underscore)가 두 개씩 추가되었다
# 이러한 메서드를 dunder(double-Underscore)라고 부르며 특별한 기능을 갖고 있다

# __repr__은 클래스 인스턴스를 문자열 형태로 반환해주는 dunder 메서드다
def __repr__(self):
    return f"CountingClikcer(count={self.count}"

# 이제 클래스를 활용할 수 있도록 퍼블릭(public) API를 만들어 보자

def click(self, num_times = 1):
    #한 번 실행할 때마다 num_times만큼 count 증가
    self.count += num_times

def read(self):
    return self.count

def reset(self):
    self.count = 0

# assert를 사용하여 테스트 조건을 만들어 보자

clicker = CoountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clikcer.click()
assert clicker.read() == 2, " after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, cliker should be back to 0"

# 이러한 테스트를 통해 작성한 코드가 정상적으로 실행되는지 확인할 수 있다
# 또한 부모 클래스에서 기능을 상속받을 수 있는 서브클래스(subclass)를 종종 사용할 것이다
# 예를 들어 CountingClicker를 상속받지만 reset 메서드를 오버라이딩(overriding)하여 
# count를 재설정할 수 없는 서브클래스를 만들 수도 있다

# 부모 클래스의 모든 기능을 상속받는 서브클래스
class NoRsetClicker(CountingClikcer):
    # CountingClicker와 동일한 메서드를 포함
    # 하지만 reset 메서드는 아무런 기능이 없도록 변경된다
    def reset(self):
        pass
clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read == 1
clicker2.reset()
assert clicker2.read() == 1, "reset shouldn't do anything"
'''
