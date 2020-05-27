#################05월 20일###################
# 3. dictionary (list 다음으로 중요함)
# 중복 X
# {키 : 밸류}
# {key : value}    
# key와 value가 쌍으로 되어있다               # key : 호출, value : 값

a = {1 : 'hi', 2 : 'hello'}                 # 1이라는 key에는 항상 'hi'가 들어가있다 (index 기능)
print(a)                                    # {1: 'hi', 2: 'hello'}
print(a[1])                                 # 출력 : hi / a의 key 1의 vlaue가 출력                

b = {'hi': 1, 'hello' : 2}                  # hi라는 key에 1, hello라는 key에 2의 값이 들어있다
print(b['hello'])                           # 출력 : 2

# 중복은 안 되나 삽입, 삭제, 수정은 가능하다

# dictionary 요소 삭제
del a[1]
print(a)                                    # 출력 : {2: 'hello'} 
                                            # key와 value는 쌍이므로 1과 hi가 함께 다 삭제된다
del a[2]
print(a)                                    # 출력 : {} / 빈 dictonary만 출력된다

a = {1 : 'a', 1 : 'b', 1 : 'c'}             # key가 중복이라면?
print(a)                                    # 출력 : {1: 'c'} / key는 중복 X
                                            # 1이라는 key에 a였다가 b로 덮어썼다가 c가 마지막에 덮어썼다고 생각
b = {1 : 'a', 2 : 'a', 3 : 'a'}             # value가 중복이라면?
print(b)                                    # 출력 : {1: 'a', 2: 'a', 3: 'a'} / Value는 중복 O, 그냥 값이므로

a = {'name' : 'ha', 'phone' : '010', 'birth' : '0717'}  
                                            
                                            # key에는 정수형, 문자형 모두 들어갈 수 있다
print(a.keys())                             # 출력 : dict_keys(['name', 'phone', 'bith']) / key값만
print(a.values())                           # 출력 : dict_values(['yun', '010', '0511']) / value만
print(type(a))                              # 출력 : <class 'dict'>

print(a.get('name'))                        # 출력 : ha
print(a['name'])                            # 출력 : ha

print(a.get('phone'))                       # 출력 : 010
print(a['phone'])                           # 출력 : 010


#개발자가해야할가장중요한코딩#
#조건문
#반복문

# 컴퓨터는 모든 연산을 더하기로 한다
# '10 - 1'을 컴퓨터는 '10 + 1의 보수'로 계산

# 조건문과 반복문만 제대로 할 줄 알면 그만
# 각각의 언어들의 문법은 모두 다르지만 그 중에서
# 같은 맥락으로 돌아가는 게 조건문과 반복문 (알고리즘이 거의 동일)

# 수업시간 강의 안 들린다고 찾아옴 -> early stopping 상황