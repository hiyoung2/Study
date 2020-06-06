# 5.5.2 리스트 안의 리스트 루프
# 리스트의 요소가 리스트형일 경우 그 내용을 for문으로 꺼낼 수 있다
# 이 때 'for a, b, c, ... in 변수 (리스트형)'과 같이 쓴다
# a, b, c, ...의 개수는 리스트의 요소 수와 같아야 한다

list = [[1, 2, 3], [4, 5, 6]]
for a, b, c in list :
    print(a, b, c)
'''
1 2 3
4 5 6
'''
# [4, 5, 6]에서 6을 빼고 실행 했더니
# ValueError: not enough values to unpack (expected 3, got 2)
# 에러 메세지 발생
# 리스트 안의 요소수가 같은 리스트들일 때만 가능

# 문제 : for문을 사용하여 다음을 출력하는 코드 작성
# - straberry is red
# - peach is pink
# banana is yellow

fruits = [["strawberry", "red"], ["peach", "pink"], ["banana", "yellow"]]

for name, color in fruits :
    print(name + " is " + color)

# strawberry is red
# peach is pink
# banana is yellow

