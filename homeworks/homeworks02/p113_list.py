# 5.1 리스트
# 5.1.1 리스트형(1)

# 앞 장에서는 변수에 하나의 값만 대입했지만, 이 장에서는 변수에 여러 값을 대입할 수 있는 리스트형(list형) 변수를 소개하겠다
# 리스트형은 수치, 문자열 등의 데이터를 한꺼번에 저장할 수 있는 자료형이며 
# [요소1, 요소2, ...] 처럼 기술한다
# 또한 리스트에 저장되어 있는 값 하나하나를 요소 또는 객체(오브젝트, object)라고 한다
# ( 삼성 주가 예측 테스트 볼 때 '시가' = object, '종가' = object.... 라는 출력화면을 봤었다)

# 다른 프로그래밍 언어를 접한 적이 있는 사람이라면 배열처럼 생각하면 좋을 것이다(접한 적이 없어요,,)

# ['코끼리', '기린', '팬더']
# [1, 5, 2, 4]

# 문제
# 변수 c에 'red', 'blue', 'yellow' 세 개의 문자열을 저장
# 변수 c의 자료형을 print 함수를 이용해 출력

c = ['red', 'blue', 'yellow']
print(c) # ['red', 'blue', 'yellow']
print(type(c)) # <class 'list'>

# 5.1.2 리스트형(2)
# 지금까지는 리스트형에 저장되어 있는 요소가 모두 같은 자료형이었지만,
# 다른 자료형이 섞여 있어도 괜찮다
# 또한 변수를 리스트 안에 저장할 수도 있다

n = 3
print(["사과", n, "고릴라"]) # ['사과', 3, '고릴라']

# 문제 : 리스트형 fruits 변수를 만들어 apple, grape, banana 변수를 요소로 저장하라

apple = 4
grape = 3
banana = 6

fruits = [apple, grape, banana] # apple, grape, banana라는 각각의 변수들이 리스트 안에 저장되었다
print(fruits) # [4, 3, 6] # 변수 안에 저장해 놓은 값들이 출력 된다

# 5.1.3 리스트 안의 리스트
# 리스트의 요소로 리스트형을 저장할 수 있다
# 즉, 중첩된 구조를 만들 수 있다

print([[1, 2], [3, 4], [5, 6]]) # [[1, 2], [3, 4], [5, 6]]

# 문제 : 변수 fruits는 '과일 이름'과 '개수' 변수를 가진 리스트이다
# [["사과", 2], ["귤", 10]]이 출력되도록 fruist에 변수를 리스트형으로 대입하라

fruits_name_1 = "사과"
fruits_name_2 = 2
fruits_name_3 = "귤"
fruits_name_4 = 10

print([[fruits_name_1, fruits_name_2], [fruits_name_3, fruits_name_4]])
# [['사과', 2], ['귤', 10]]

# 5.1.4 리스트에서 값 추출
# 리스트 요소는 차례대로 0, 1, 2, 3, ... 이라는 번호가 할당되어 있다
# 이를 인덱스 번호라고 한다
# 인덱스 번호는 0부터 시작하므로 첫 번째 요소가 0번째라는 점에 항상 주의!
# 또한 리스트 요소는 뒤에서부터 순서대로 번호를 지정할 수도 있다

# 가장 마지막 요소는 -1번째, 끝에서 두 번째 요소는 -2번째처럼 지정할 수 있다
# 리스트의 각 요소는 '리스트[인덱스 번호]'로 검색할 수 있다

a = [1, 2, 3, 4]
print(a[1])  # 2 
print(a[-2]) # 3

# 문제 
# - 변수 fruits의 두 번째 요소를 출력
# - 변수 fruits의 마지막 요소를 출력
# - 출력은 print() 함수를 사용

fruits = ['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1]
print(fruits[1])  # 2
print(fruits[-1]) # 1

# 5.1.5 리스트에서 리스트 추출(슬라이스)
# 리스트에서 새로운 리스트를 추출할 수도 있다
# 이 작업을 슬라이스라고 한다
# 작성법은 '리스트[start:end]'이며, 인덱스 번호 start부터 end-1 까지 리스트를 출력한다

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
print(alphabet[1:5])  # ['b', 'c', 'd', 'e']
print(alphabet[1:-5]) # ['b', 'c', 'd', 'e']
print(alphabet[:5])   # ['a', 'b', 'c', 'd', 'e']
print(alphabet[6:])   # ['g', 'h', 'i', 'j']
print(alphabet[0:20]) # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# 20번째 idnex는 없어도 자동으로 끝부분까지 출력되는군
# 6: -> index 6번부터 끝까지 출력

# 문제 
# - chaos 리스트에서 다음 리스트를 꺼내 변수 fruits 에 저장하라
# - ["apple", 2, "orange", 4, "grape", 3, "banana", 1]
# - 변수 fruits를 print() 함수로 출력

chaos = ["cat", "apple", 2, "orange", 4, "grape", 3, "banana", 1, "elephant", "dog"]
fruits = chaos[1:9]
print(fruits) # ['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1]

# 5.1.6 리스트 요소 갱신 및 추가
# 리스트 요소(객체)를 갱신하거나 추가할 수 있다
# '리스트[인덱스 번호] = 값'을 사용하면 지정한 인덱스 번호의 요소를 갱신할 수 있다
# 슬라이스를 이용하여 값을 갱신할 수도 있다
# 리스트에 요소를 추가하고 싶은 경우, 리스트와 리스트를 '+'를 사용하여 연결한다
# 여러 여소를 동시에 추가하는 것도 가능하다
# '리스트명.append(추가 요소)'로 추가할 수도 있다
# append () 메서드를 사용할 경우, 여러 요소를 동시에 추가할 수 있다

alphabet = ["a", "b", "c", "d", "e"]
alphabet[0] = "A"
alphabet[1:3] = ["B", "C"]
print(alphabet) # ['A', 'B', 'C', 'd', 'e']

alphabet = alphabet + ["f"] # +를 사용해 더해줄 때에는 []안에 넣어서 더함
alphabet += ["g", "h"]
alphabet.append("i") # append 할 때는 (), 소괄호를 쓴다
print(alphabet) # ['A', 'B', 'C', 'd', 'e', 'f', 'g', 'h', 'i']

# 문제
# - 리스트 c의 첫 요소를 'red'로 갱신
# - 리스트 끝에 문자열 'green'을 추가

c = ["dog", "blue", "yellow"]
c[0] = "red"
print(c) # ['red', 'blue', 'yellow']

c.append("green")
print(c) # ['red', 'blue', 'yellow', 'green']

# 5.1.7 리스트 요소 삭제
# 리스트 요소를 삭제하려면 'del 리스트[인덱스 번호]'라고 기술한다
# 그러면 지정한 인덱스 번호의 요소가 삭제된다
# 인덱스 번호를 슬라이스로 지정할 수도 있다

alphabet = ["a", "b", "c", "d", "e"]
del alphabet[3:]
del alphabet[0]
print(alphabet) # ['b', 'c']

# 문제 : 변수 c의 첫 번째 요소를 제거하라
c = ["dog", "blue", "yellow"]
print(c)

del c[0] # ['blue', 'yellow']

# 5.1.8 리스트형의 주의점
alphabet = ["a", "b", "c"]
alphabet_copy = alphabet
alphabet_copy[0] = "A"
print(alphabet) # ['A', 'b', 'c']

# 리스트 변수를 다른 변수에 대입한 뒤, 대입한 변수에서 값을 바꾸면 원래 변수의 값도 바뀐다
# 이를 막으려면  y = x 대신 y = x[:] 또는 y = list(x)라고 쓰면 된다

alphabet = ["a", "b", "c"]
alphabet_copy = alphabet[:] # 또는 alphabet_copy = list(alphabet)
alphabet_copy[0] = "A"
print(alphabet)      # ['a', 'b', 'c']
print(alphabet_copy) # ['A', 'b', 'c']

# 문제 : 변수 c의 리스트 요소가 변하지 않도록 'c_copy = c' 부분을 수정하라

c = ["red", "blue", "yellow"]

c_copy = c[:] # 또는 c_copy = list(c)

c_copy[1] = "green"
print(c) # ['red', 'blue', 'yellow'] / c 리스트의 요소들은 변하지 않았다
print(c_copy) # ['red', 'green', 'yellow'] / c_copy 리스트의 인덱스 1번 요소가 바뀌었다
