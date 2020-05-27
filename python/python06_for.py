# 반복문 for
"""
옛날 문법
a = 0
for ( i = 1, i = 100, i ++){
    a = a + i
}
print(a)

a i 결과a
0 1 1
1 2 3
3 3 6 
6 4 10
...
  100 a+100
"""

# 자동 Tap 되는 것은 블록 설정이 되었다는 뜻
# for문 입력하고 : 입력 후 엔터를 하면 자동 Tap이 된다
# for문의 나와바리 설정 된 것

############################################################################
# for the moon by the sea 네가 떠난 바닷가에 눈물이 마를 때까지(다 마를 때까지)
# 사랑한다는 건 오직 기다림뿐이었단 걸 난 왜 몰랐을까
# 내 꿈엔 비가 이젠 그치길
############################################################################


# python 문법
# for i in 100:  # 100개 하나씩 늘려라


a = {'name' : 'ha', 'phone' : '010', 'brith' : '0717'}  

for i in a.keys():            # a.keys 값들을 i에 하나씩 넣어준다
    print(i)                  # name
                              # phone
                              # brith    
                              # 위와 같이 key값들이 순서대로 출력 된다

                              # i 정의하지 않았는데 어떻게 실행이 되는 건가?
                              # i 대신 아무거나 집어 넣어도 실행 가능?

# for inyoung in a.keys():      이런 식으로 임의의 변수명을 넣어도 출력 결과는 같다   
    # print(inyoung)            for 안에서 이미 정의가 되었기 때문에 따로 정의할 필요가 없다

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:                   # in 뒤에 리스트형이거너 무엇이건 들어갈 수 있다(?)
    i = i*i
    print(i)                  # 1, 4, ..., 100 / 10개의 값 출력
    # print('melong')         # 이렇게 되면 melong이 10번 출력 된다 (for문 나와바리에 있기 때문에)
# print('melong')             # for문 나와바리 밖이므로 그냥 1번만 출력 된다    

for i in a:
    print(i)                  # 1, 2, 3, ..., 10 / 10개의 값 출력

# whiile문
'''
while 조건문 :                 -> 참일 동안 계속 돈다
    수행할 문장
''' 

# if문
'''
if 조건문 :
    print(조건문이 맞을 경우 출력되는 것)
else 
    print(if 조건문이 충족되지 않을 경우 출력되는 것)
'''


if 1:                         # 1이 True이므로 True가 출력
    print('True')
else: 
    print('False')             # 출력 : True
                               # 1이면 True 아니면 False 

if 3: 
    print('True')
else: 
    print('False')             # 출력 : True                         

if 0:
    print('True')
else: 
    print('False')             # 출력 : False
 
if -1:
    print('True')
else:
    print('False')             # 출력 : True
 

'''
비교연산자

<, >, ==, !=, >=, <=

'''

a = 1
if a == 1:
    print('출력 잘 돼')         # a = 1 이라고 조건문에 입력하면
                               # SyntaxError: invalid syntax / 문법오류
                               # a = 1 : a에 1을 대입하는 것
                               # 조건문 성립하지 않음, 따라서 a == 1 이라는
                               # 조건 연산자 '=='을 사용해야 한다

money = 10000
if money >=30000:
    print('한우 먹자')
else: 
    print('라면 먹자')          # 출력 : 라면 먹자

# 조건문, 들여쓰기(Tap)영역 #

# 조건연산자
# and, or, not

money = 20000
card = 1

if money >= 30000 or card == 1:
    print("한우 먹자")
else:
    print("라면 먹자")          # 출력 : 한우 먹자


print("============================================")


jumsu = [90, 25, 67, 45, 80]    # 점수 : 리스트
number = 0

for i in jumsu:                 # i에 리스트에 들어있는 점수(인자)가 하나씩 들어간다(하나씩 반복 실행) : 총 5번 돈다
    if i >= 60:     
        print("경] 합격 [축")
        number = number + 1
print("합격인원 : ", number, "명")  

'''
출력 
경] 합격 [축
경] 합격 [축
경] 합격 [축
합격인원 :  3 명

처음에 첫 번째 인자 90이 i로 들어간다, i>=60 충족되니까 print문 실행 되고 number 0 = 0+1 -> 1이 됨
두 번째 25가 i로 들어감 60보다 작으니까 아래 전체가 실행 되지 않음
세 번째 67 실행 됨 print 실행 되고 1 = 1+1 -> 2가 됨
45 실행X
80 실행 됨, print 실행 되고 2 = 2+1 3이 됨
number에는 총 3이 들어감
결과 : 경] 합격 [축 3번 출력 되고 합격인원 3명 출력 된다
'''

##########################################################
# break, continue

print("=================break====================")


jumsu = [90, 25, 67, 45, 80] 
number = 0
for i in jumsu:
    if i < 30:
        break 
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 : ", number, "명")  

'''
출력
경] 합격 [축
합격인원 :  1 명

90 30보다 작지 않고 60보다 크니까 프린트 돌고 넘버 1추가
25는 30보다 작으니까 조건 충족, break문에 걸림, break 걸리면 브레이크문에서 가장 가까운 for문이 실행 중지 된다
for문이 여러 개 있을 경우 가장 가까운 for문을 뻥하고 날려버림
for문을 탈출하고 for문 나와바리 아닌 print 실행 된다
'''

print("==============continue====================")

jumsu = [90, 25, 67, 45, 80] 
number = 0
for i in jumsu:
    if i < 60:
        continue
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 : ", number, "명")  

'''
출력
경] 합격 [축
경] 합격 [축
경] 합격 [축
합격인원 :  3 명

90은 30보다 큼, 진행 x, 25는 작음 -> 하단 부분 아예 실행 X, 
'''

# break를 더 많이 씀