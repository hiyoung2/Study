# list 가장 중요하다. python은 list 위주로 공부

# 자료형
# 1. 리스트, list

# list는 여러가지 자료형 같이 쓸 수 있다. 
# 삽입, 삭제, 수정 모두 가능하다. 
# 대괄호 안에 넣어주는 것이 문법


a = [1,2,3,4,5]        
b = [1,2,3, 'a', 'b']            # [] 안에 ''도 올 수 있음 , 공백이 올 수 있음(공백도 문자)
                                 # int, str 등 여러 형태를 같이 사용 가능하다 

print(b)                         # 출력 : [1, 2, 3, 'a', 'b']     

print(a[0] + a[3])
# print(b[0] + b[3])             # TypeError: unsupported operand type(s) for +: 'int' and 'str'

print(type(a))                   # <class 'list'>
print(a[-2])                     # 4 
print(a[1:3])                    # [2,3] 

a = [1, 2, 3, ['a', 'b', 'c']]   
                                 # 중첩(리스트 안에 리스트)
print(a[1])
print(a[-1])                     # 출력 : ['a', 'b', 'c']
print(a[-1][1])                  # 출력 : b
                                 # [-1] list 안의 [1]이니까

# 1-2. 리스트 슬라이싱
a = [1, 2, 3, 4, 5]
print(a[:2])                     # [1, 2]
print(a[1:2])
####### 항상 생각해보고 실행시키기

# 1-3. 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)                     # 리스트 한 덩어리 + 한 덩어리, 그냥 단순 리스트 병합
                                 # 출력 : [1, 2, 3, 4, 5, 6]
                                 # if, [5, 7, 9] 라는 결과를 얻고 싶다면...? -> numpy  


# a 를 wx, b 를 b라고 생각하면,
# 현재 print(a+b)는 shape만 더 커지고 연산은 안 이뤄짐
# 우리는 연산이 필요
# 해결방법 : numpy
# numpy를 써주면 위의 리스트들의 덧셈이 우리가 원하는 방식으로 출력이 된다
# numpy의 단점 : 같은 type끼리만 가능
# ***keras14_mlp.py 참고

c = [7, 8, 9, 10]
print(a + c)
print(a * 3)                      # 출력 : [1, 2, 3, 1, 2, 3, 1, 2, 3]
                                  # numpy를 쓰면 [3, 6, 9] 로 출력


###### 인공지능에 쓰이는 파이썬 기본 배우는 중...

# print(a[2] + 'hi')              # TypeError: unsupported operand type(s) for +: 'int' and 'str'
                                  # a는 list 형, a[2]는 int(3), hi는 str
print(str(a[2]) + 'hi')           # a[2]를 str로 형 변환하면 오류 해결

f = '5'
# print(a[2]+f)                   # TypeError: unsupported operand type(s) for +: 'int' and 'str'      
print(a[2] + int(f))

# 파이썬 list 교보문고 가서 읽어보기 
# 리스트 관련 함수
a = [1, 2, 3]
a.append(4)                        # data가 추가 되었을 때 쓸 수 있다, 정말 많이 쓴다
                            
print(a)                           # 출력 : [1, 2, 3, 4]

                                   # .append 쓸 때 주의 사항 
                                   # a = a.append(5) NONO
# print(a)                         # 출력 : none (오류)
                            # .append는 a = 식으로 써 주지 않아도 된다

a = [1, 3, 4, 2]
a.sort()
print(a)                           # sort : 정렬, 출력 : [1, 2, 3, 4] 
                                   # .sort 역시 a = a.sort 이렇게 쓰지 않는다
a.reverse()                        # 역순으로 정렬
print(a)                           # 출력 : [4, 3, 2, 1]

# index : 색인
print(a.index(3))                  # == a[3] / 출력 : 1
print(a.index(1))                  # == a[1] / 출력 : 3

# insert : 삽입             
a.insert(0, 7)                     # 0번째에 7을 삽입하겠다 (교체가 아님!)
print(a)                           # 출력 : [7, 4, 3, 2, 1]

a.insert(3, 3)                     # 3번째에 3을 삽입(뒤의 나머지 숫자는 뒤로 밀림)
print(a)                           # 출력 : [7, 4, 3, 3, 2, 1]

a.remove(7)                        # remove : 삭제 / remove 안에 있는 인자를 삭제
print(a)                           # 출력 : [4, 3, 3, 2, 1]

a.remove(3)                        # 먼저 걸리는 놈만 지워진다 (3이 두 개 있지만 앞의 3만 remove)
print(a)                           # 출력 : [4, 3, 2, 1]


# append, sort, reverse, index, insert, remove 모두 a.~~ 형태로 사용 / a = a.~~~ 으로 쓰면 error
# append 정말 많이 쓰이고 가장 중요, 잊지 말아야 할 함수!!!

# 조원이 한 명 추가, append를 쓴다!
# 수업 내내 나올 append
# a.append()는 무조건 손에 익혀라.

####################################################################################
###### list는 정말 중요, 꼭 공부하기, 그 중 slice, indexing, append 항상 나온다. ######
####################################################################################