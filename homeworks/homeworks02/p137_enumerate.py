# 5.5 추가 설명
# 5.5.1 for문에서 index 표시

# for문을 사용한 루프에서 리스트의 인덱스 확인이 필요할 떄가 있다
# enumerate() 함수를 사용하면 인덱스가 포함된 요소를 얻을 수 있다

# 형식
'''
for x, y in enumerate("리스트형") :
    for 안에서는 x, y를 사용하여 작성한다
    x는 정수형 인덱스, y는 리스트에 포함된 요소이다
'''
# x, y는 인덱스와 요소를 얻기 위한 변수이며, 자유롭게 이름을 붙일 수 있다
list = ["a", "b"]

for index, value in enumerate(list) :
    print(index, value)
'''
0 a
1 b
'''

# 문제
# - for문 및 enumerate() 함수를 사용하여 다음을 출력하는 코드를 작성
# - index: 0 tiger
# - index: 1 dog
# - index: 2 elephant
# - 출력은 print() 함수 
animals = ["tiger", "dog", "elephant"]

for index, value in enumerate(animals) :
    print("index:"+str(index), value)
'''
index:0 tiger
index:1 dog
index:2 elephant
'''

