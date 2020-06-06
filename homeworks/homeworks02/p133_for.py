# 5.4 for문
# 5.4.1 for문

# 리스트의 요소를 모두 출력하고 싶을 때 자주 사용하는 것이 for문이다
# 'for 변수 in 데이터셋 : '이라고 작성하면 데이터셋의 요소 수큼 반복처리 할 수 있다

# 데이터셋은 리스트형이나 딕셔너리형처럼 복수의 요소를 가진 것을 가리킨다
# for문에 리스트형을 사용해보자
# for문 뒤에는 콜론, :이 들어가는 것을 잊지 말자
# if, while 과 마찬가지로 들여쓰기로 처리 범위를 나타낸다
# 들여쓰기는 공백 4개

# for문 예시
animals = ["tiger", "dog", "elephant"] # animals라는 리스트 형식의 데이터셋 생성
for animal in animals : # animals 데이터셋에서 하나하나 요소를 가져오는데, 변수 animal로 지정
    print(animal)
'''
tiger
dog
elephant
'''

# 문제 
# - for문을 사용하여 변수 sotrages의 요소를 하나씩 출력
# - 출력은 print() 함수를 사용
# - for문에서 사용할 변수명은 임의로 지정

storages = [1, 2, 3, 4]
for num in storages : 
    print(num)

'''
1
2
3
4
'''

# 5.4.2 break
# break를 이용해서 반복 처리를 종료할 수 있다
# if문과 함께 사용되는 경우가 많다

storages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in storages :
    print(n)
    if n >= 5 :
        print("끝")
        break
'''
1
2
3
4
5
끝
'''

# 문제 : 변수 n의 값이 4일 때 처리를 종료하시오
storages = [1, 2, 3, 4, 5 , 6]

for n in storages :
    print(n)
    if n == 4:
        break
'''
1
2
3
4
'''

# 5.4.3 continue
# continue는 break와 마찬가지로 if문 등과 조합해서 사용하지만 break와 달리 특정 조건일 때 루프를 한 번 건너뛴다

storages = [1, 2, 3]
for n in storages :
    if n == 2 :
        continue
    print(n)
'''
1
3
'''

# 문제 : 변수 n이 2의 배수일 때는 continue를 사용하여 처리를 생략
storages = [1, 2, 3, 4, 5, 6]

for n in storages :
    if n % 2 == 0 :
        continue
    print(n)
'''
1
3
5
'''

