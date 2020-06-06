# 6. 함수 기초
# 6.1 내장 함수와 메서드
# 6.1.1 내장 함수

# 함수는 간단히 말해 처리를 정리한 프로그램
# 사용자는 함수를 자유롭게 정의할 수 있으며, 여러 함수가 들어있는 패키지도 있다
# 이러한 패키지는 라이브러리, 프레임워크 등으로 불린다

# 내장함수란 파이썬에 미리 정의된 함수이며, 대표적인 예로 print() 함수가 있다
# 파이썬은 print() 외에도 많은 편리한 내장 함수가 준비되어 있다
# 이들을 이용하여 효율적으로 프로그램을 작성 할 수 있다
# print(), type(), int(), str() 등도 내장 함수이다

# len()에 대해 알아볼 것
# len() 함수는 ()내 객체의 길이나 요소 수를 돌려준다

# 객체(오브젝트)는 변수에 할당할 수 있는 요소를 말한다
# 대입되는 값은 인수라고 한다
# 인술르 파라미터라고 부르는 경우도 있다

# 함수에서 인수로 받는 변수의 자료형은 정해져 있다
# len() 함수는 문자열형(str형)과 리스트형(list형)을 인수로 받을 수 있지만
# 정수형(int형), 부동소수점형(float형), 불리언형(bool형) 등은 인수로 받을 수 없다
# 함수를 익힐 때는 어떤 자료형의 인수를 사용할 수 있는지 확인해두자
# 인수를 확인할 때는 파이썬의 레퍼런스를 참조하는 것이 좋다

# 인수에 따라 오류가 발생하지 않는 예와 발생하는 예
print(len("tomato"))  # 6
print(len([1, 2, 3])) # 3

# 오류가 발생하는 예
# print(len(3))       
# TypeError: object of type 'int' has no len()

# print(len(2.1))
# TypeError: object of type 'float' has no len()

# print(len(True))
# TypeError: object of type 'bool' has no len()

# 문제
# - 변수 vege의 객체 길이를 len() 함수와 print() 함수를 이용하여 출력
# - 변수 n의 객체 길이를 ~

vege = 'potato'
n = [4, 5, 2, 7, 6]

print(len(vege))  # 6
print(len(n))     # 5

# 6.1.2 메서드
# 메서드는 어떠한 값에 대해 처리를 하는 것이며, '값.메서드명()' 형식으로 기술한다
# 역할은 함수와 동일하다
# 그러나, 함수의 경우 처리하려는 값을 () 안에 기입했지만, 메서드는 값 뒤에 .(점)을 연결해 기술한다는 점을 기억
# 함수와 마찬가지로 값의 자료형에 따라 사용할 수 있는 메서드가 다르다
# 예를 들어, append()는 리스트형에 사용할 수 있는 메서드이다

# append 복습
alphabet = ["a", "b", "c", "d" , "e"]
alphabet.append("f") # method 사용법! 값.메서드명()
print(alphabet)
# ['a', 'b', 'c', 'd', 'e', 'f']

# 내장함수와 메서드가 같은 처리를 제공하는 경우도 있다
# 내장 함수 sorted()와 메서드 sort()를 예로 들 수 있다
# -> 정렬용 함수와 메서드이다

# sorted -> 내장함수
number = [1, 5, 3, 4, 2]
print(sorted(number))  # [1, 2, 3, 4, 5]
print(number)          # [1, 5, 3, 4, 2]

# sort -> method
number = [1, 5, 3, 4, 2]
number.sort()
print(number) # [1, 2, 3, 4, 5]

# 같은 정렬용이지만, print(number)를 할 때, 원래 값이 변화했는가? 하는 점에서 차이가 난다
# 변수의 내용을 변경하지 않은 것이 sorted() 이며, 변수의 내용까지 변경해버린 것이 sort()이다
# but! 모든 내장 함수와 메서드가 이러한 관계가 성립하는 것은 아니다
# 이처럼 원래 리스트의 내용 자체를 바꿔버리는 메서드인 sort()는 프로그래밍 세계에서
# 파괴적 메서드, destructive method 라고 부르기도 한다

# 6.1.3 문자열형 메서드(upper, count)
# 문자열형 메서드형인 upper()와 count()
# upper()는 모든 문자열을 대문자로 반환하는 메서드이다
# count()는 () 안에 들어 있는 문자열에 요소가 몇 개 포함되어 있는지 알려주는 메서드이다
# 사용법은 각각 '변수.upper()'와 '변수.count("객체")'이다

city = "Seoul"
print(city.upper())    # SEOUL
print(city.count("o")) # 1

# 변수 animal에 저장되어 있는 문자열을 대문자로 변환해서 변수 animal_big에 저장
# 변수 animal에 'e'가 몇 개 포함되어 있는지 출력

animal = "elephant"

animal_big = animal.upper()
print(animal)     # elephant
print(animal_big) # ELEPHANT
print(animal.count("e")) # 2

# 6.1.4 문자열형 메서드(format)
# 문자열형에는 유용한 format() 메서드가 있다
# format() 메서드는 임의의 값을 대입한 문자열을 생성할 수 있다
# 문자열에 변수를 삽입할 때 자주 사용한다
# 문자열 내에 {}를 포함하는 것이 특징
# {} 안에 값이 들어간다

print("나는 {}에서 태어나 {}에서 유년기를 보냈다.".format("포항", "경주"))
# 나는 포항에서 태어나 경주에서 유년기를 보냈다.

# 문제
# format() 메서드를 사용하여 '바나나는 노란색입니다.' 라고 출력

fruit = "바나나"
color = "노란색"

print("{}는 {}입니다.".format(fruit, color))
# 바나나는 노란색입니다.

# 6.1.5 리스트형 메서드(index)
# 리스트형에는 인덱스 번호가 존재한다
# 인덱스 번호는 리스트 내용을 0부터 순서대로 나열했을 때의 번호이다
# 객체의 인덱스 번호를 찾는 용도의 index() 메서드가 제공된다
# 리스트형에서도 앞서 다룬 count() 메서드를 사용할 수 있다

alphabet = ["a", "b", "c", "d", "d"]
print(alphabet.index("a")) # 0 / 0번째 index
print(alphabet.count("d")) # 2 / "d"는 2개가 있다

# 문제
# - '2'의 인덱스 번호를 출력
# - 변수 n에 '6'이 몇 개 들어있는지 출력

n = [3, 6, 8, 6, 3, 2, 4, 6]

print(n.index(2)) # 5
print(n.count(6)) # 3

# 리스트 요소들이 문자열 아니고 그냥 정수형이므로
# ()안에 "2"나 "6"이 아니라 그냥 2, 6을 쓰면 된다


# 6.1.6 리스트형 메서드(sort)
# sort() 메서드는 리스트형에서 자주 사용
# sort() 메서드는 리스트를 오름차순으로 정렬한다
# reverse() 메서드를 사용하면 리스트 요소를 반대로 할 수 있다
# sort() 메서드를 사용하면 리스트의 내용이 변경된다
# 그러므로 단순히 정렬된 리스트를 보고 싶을 뿐이라면, 내장 함수인 sorted()를 사용하는 것이 좋다

# sort() method의 예
list = [1, 10, 2, 20]
list.sort()
print(list) # [1, 2, 10, 20]

# reverse() 메서드의 예
list = ["가", "나", "다", "라", "마"]
list.reverse()
print(list) # ['마', '라', '다', '나', '가']

# 문제
# - 변수 n을 정렬하여 오름차순으로 출력
# - n.reverse()로 정렬된 변수 n의 요소를 순서를 반대로 하여 내림차순으로 출력

n = [53, 26, 37, 69, 24, 2]
n.sort()
print(n) # [2, 24, 26, 37, 53, 69]
n.reverse()
print(n) # [69, 53, 37, 26, 24, 2]

# 6.2 함수
# 6.2.1 함수 작성
# 함수는 프로그램의 여러 처리를 하나로 정리한 것이다
# 정확하게는 인수를 받아 처리한 결과를 반환값으로 돌려준다
# 함수를 사용하면 전체적인 동작이 알게 쉬워지고, 동일한 처리를 여러 번 작성하지 않아도 된다는 장점이 있다
# -> 재사용이 가능하다는 말!
# 함수의 작성법은 'def 함수명(인수):'
# 인수는 함수에 전달하려는 값이다
# 인수가 비어 있는 경우도 있다
# 함수의 처리 범위는 역시나 들여쓰기

# 함수를 호출할 때는 '함수명()'을 사용한다
# 함수는 정의한 후에만!!! 호출할 수 있다

# 다음은 인수가 없는 간단한 함수
# 함수 작성법과 호출 방식을 확인해보자

def sing():
    print("노래합니다!~~")

sing() # 노래합니다!~~

# 문제 : '홍길동입니다.' 라고 출력하는 함수 introduce를 작성

def introduce():
    print("홍길동입니다.")

introduce() # 홍길동입니다.

# 6.2.2 인수
# 함수에 전달하는 값을 '인수'라고 했다
# 함수는 인수를 받아서 그 값을 사용한다

# 'def 함수명(인수)'처럼 인수를 지정한다
# '함수명(인수)'로 함수를 호출할 때, 전달된 인수(값)가 인수로 지정한 변수에 대입되기 때문에
# 인수를 바꾸는 것만으로 출력 내용을 변경할 수 있다
# 인수와 함수에 정의된 변수는 그 함수 내에서만 사용할 수 있다는 점에 주의

# 다음은 인수가 하나인 함수이다
# 함수 작성법과 호출 방식 확인해보자

def introduce_myself(n) :
    print(n + "입니다.")

introduce_myself("하인영") # 하인영입니다.

# 문제 : 인수 n을 세제곱한 값을 출력하는 함수 cube_cal 을 작성

def cube_cal(n) :
    print(n**3)

cube_cal(4) # 64

# 6.2.3 복수 개의 인수
# 복수 개의 인수를 전달하려면 () 안에 쉼표로 구분하여 지정한다
# 다음은 인수를 두 개 지정한 함수이다

def introduce_2(first, second) :
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")

introduce_2("하", "인영") # 성은 하이고, 이름은 인영입니다.

# 인수 n과 age를 이용하여 '**입니다. **살입니다.'를 출력하는 함수 introduce를 작성
# '홍길동'과 '18'을 인수로 하여 함수 introduce를 호출
# '홍길동'은 문자열, '18'은 정수로 지정

def introduce_3(n, age) :
    print(n+ "입니다. " + str(age) + "살입니다.")

introduce_3("홍길동" , 18) # 홍길동입니다. 18살입니다.

# 6.4.2 인수의 초깃값
# 인수에 초깃값을 설정할 수 있다
# 초깃값을 설정하면 '함수명(인수)'로 호출 시 인수를 생략하면 초깃값이 사용된다
# 초깃값을 설정하려면 () 안에 '인수=초깃값' 이라고 적으면 된다

def introduce_4(first = "진", second = "유나") :
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")

introduce_4("강") # 성은 강이고, 이름은 유나입니다.

# 초깃값을 설정한 인수 뒤에 초깃값을 설정하지 않는 인수를 둘 수 없다는 점에 주의
# 다음은 가능하지만
def introduce_5(first, second = "유나") :
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")

introduce_5("강") # 성은 강이고, 이름은 유나입니다.

# 다음은 가능하지 않다
# def introduce_6(first = "강", second) :
#     print("성은 " + first + "이고, 이름은 " + second + "입니다.")

#   File "d:\Study_young\assignment\python_dl.py\p145_ftn_method.py", line 262
#     def introduce_6(first = "강", second) :
#                    ^
# SyntaxError: non-default argument follows default argument


# 문제 : n의 초깃값을 '홍길동', '18'만 인수로 넣어 함수 호출

def introduce_7(age, n = "홍길동") :
    print(n + "입니다. " + str(age) + "살입니다.")

introduce_7(18) # 홍길동입니다. 18살입니다.

# 6.2.5 return
# 함수는 반환값을 설정하여 함수를 호출한 곳으로 그 값을 되돌릴 수 있다
# 'return 반환값' 으로 기술한다
# return 뒤에 실행 결과를 직접 적을 수도 있다

def introduce_8(first = "김", second = "길동") :
    return "성은 " + first + "이고, 이름은 "  + second + "입니다."

print(introduce_8("홍")) # 성은 홍이고, 이름은 길동입니다.


# return 뒤에 문자가 길게 나열되면 함수를 이해하기 힘들기 때문에 변수를 정의하여 변환하는 것도 가능하다

def introduce_9(first = "김", second = "길동") :
    comment = "성은 " + first + "이고, 이름은 "  + second + "입니다."
    return comment

print(introduce_9("홍")) # 성은 홍이고, 이름은 길동입니다.


# 문제 : bmi을 계산하는 함수를 작성, bmi 값을 반환하라

def bmi(weight, height) :
    cal = weight / height**2
    return cal

print(bmi(65, 1.65))

# 6.2.6 함수 import(가져오기)
# 파이썬에서는 우리가 직접 만든 함수 외에도 공개된 함수를 import 하여 사용할 수있다
# 이런 함수는 유사한 용도끼리 set으로 공개되어 있다
# 이 set를 package라고 한다
# package 안에 들어 있는 하나하나의 함수를 module 이라고 한다


# time package를 보자

# 실행 시간의 출력이나 프로그램의 중지 시간 등 시간과 관련된 함수는 time이라는 패키지로 공개되어 있다
# 또한 time 패키지에는 프로그램에서 사용하는 모듈이 여러 개 포함되어 있다

# 패키지는 import 하여 사용할 수 있다 
# 패키지를 import 하면 '패키지명.모듈명' 으로 함수를 사용할 수 있다
# time 패키지를 import 하여 현재 시간을 출력해보자!

import time
now_time = time.time()

print(now_time) # 1591434540.4372745

# 'from 패키지명 import 모듈명' 으로 모듈을 import 하면 패키지명을 생략하고 
# 모듈명만으로 모듈을 사용할 수 있다

from time import time
now_time = time()
print(now_time) # 1591434605.7177646

# 패키지에는 어떤 종류가 있을까?
# Python 에는 PyPI(Python Package Index) 라는 패키지 관리 시스템이 있으며
# 그곳에 공개되어 있는 패키지를 자신의 컴퓨터에 설치해서 사용할 수 있다
# PyPI에서 패키지를 다운로드 하는 관리 도구로 pip가 잘 알려져 있다
# 명령 프롬프트(Windows 이외에는 터미널) 에서 'pip install 패키지명'을 입력하여 설치할 수 있다

# 문제 : from을 이용하여 time 패키지의 time 모듈을 import 하고 time()을 이용하여 현재 시간 출력
from time import time
now_time = time()
print("what time is it now? :", now_time) # what time is it now? : 1591434768.1015525

