# 5.5.3 딕셔너리형의 루프
# 딕셔너리형의 루프에서는 키와 값을 모두 변수로 하여 반복(루프) 할 수 있다
# items()를 사용하여 'for key의_변수명, value의_변수명 in 변수(딕셔너리형).items():'로 기술한다

fruits = {"strawberry" : "red", "peach" : "pink", "banana" : "yellow"}

for fruit, color in fruits.items() :
    print(fruit + " is " + color)

'''
strawberry is red
peach is pink
banana is yellow
'''

# 문제 : for문을 사용하여 다음을 출력하는 코드를 작성
# 경기도 분당
# 서울 중구
# 제주도 제주시

town = {"경기도" : "분당", "서울" : "중구", "제주도" : "제주시"}
for t1, t2 in town.items() :
    print(t1, t2)
'''
경기도 분당
서울 중구
제주도 제주시  
'''

