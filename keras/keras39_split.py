# copy

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11))
size = 5        # lstm에서의 timesteps랑 같은 역할

# seq와 size는 매개변수
# : 내가 함수에 적용하고 싶은 변수를 대신해서 정의하는 과정에 사용된 것
# 함수를 쓰는 이유는 재사용!!!!
# 재사용을 위해서는 직접 함수에 넣어볼 변수 대신에 임시적인 이름을 가진 변수를 함수를 정의 할 때 쓴다

# 1부터 10까지 연속된 데이터로 5개씩 잘랐음 # 6 x 5 -> 결과

#### 너무나도 당연한!!!!!!!!!!!!!!!!!!!!!!python 문법
#### for i in 100:  # 100개 하나씩 늘려라
# 1-2. 리스트 슬라이싱
# a = [1, 2, 3, 4, 5]
# print(a[:2])                     # [1, 2]
# print(a[1:2])
# a = [1, 2, 3]
# a.append(4)                        # data가 추가 되었을 때 쓸 수 있다, 정말 많이 쓴다
                            
# print(a)                           # 출력 : [1, 2, 3, 4]

def split_x(seq, size):                             # split_x라는 함수를 정의하겠다, 매개변수는 seq, size 현재 이 소스에서는 seq에 a, size는 size
                                                    
    aaa = []                                        # aaa = [] list!의 역할
    for i in range(len(seq) - size + 1):            # len seq  : seq의 길이 , 이 소스에서는 a의 길이
                                                    # (10 - 5(size값, 함수 전에 size라는 상자 안에 5를 넣어둠) + 1) = 6
                                                    # range(6) : 0, 1, 2, 3, 4, 5
                                                    # i가 0 부터 5까지 들어가면서 for문 반복
        subset = seq[i : (i+size)]                  # slicing! seq = a(여기에서), 따라서 a[0:(0+5 = 5)] = [1, 2, 3, 4, 5] 
                                                    # subset이라는 변수(상자) 안에 [1,2,3,4,5]가 저장된다
        aaa.append([item for item in subset])       # aaa는 []! 리스트에 append한다, 무엇을? subset을!
        # == aaa.append(subset) 더 단순하게 표현     # 그러면 for 문 1번 실행 결과는? [[1,2,3,4,5]] 가 되겠지
                                                    # 
    print(type(aaa))                                # i가 0부터 5까지 들어가므로 6번 실행된다는 말, 6번 실행 후에는 for문 나와바리를 벗어나서 aaa 의 type 인 list가 출력
                                                    # 
    
    return np.array(aaa)                            # 최종적인 반환 값은 aaa            


dataset = split_x(a, size)                          # 함수 호출, dataset라는 변수에 split_x(a, size)의 반환값을 저장
                                                    # 위에서 지정한 a와 size 변수가 이 함수의 인자
print("=================")
print(dataset)                                      # 변수 dataset(split_x함수의 반환값) 출력


'''
i = 0
subset = a[0 : 5] ([1,2,3,4,5])
aaa.append([1,2,3,4,5]) / aaa 는 []로 설정되어있음 , 빈 리스트, 거기에 [1,2,3,4,5]를 append, 붙여준다 
그려면 현재 aaa는 [[1,2,3,4,5]]

i = 1
subset = a[1 : 6]
aaa.append([2,3,4,5,6]) / aaa는 for문 한 번의 실행으로 [1,2,3,4,5]였는데 거기에 [2,3,4,5,6]을 append, 덧붙여준다
aaa = [[1,2,3,4,5],[2,3,4,5,6]]

i = 2
susbset = a[2:7]
aaa.append([3,4,5,6,7]
aaa = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]

for문 계속 반복

i = 5 (for문은 6번 실행된다고 했으니 마지막 실행)
subset = a[5:10] ([5,6,7,8,910])
aaa.append([5,6,7,8,9,10])
aaa = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10]]

for 문 탈출 하고 print(type(aaa)), aaa의 타입 출력한 후, aaa를 함수값으로 반환한다


'''


# a = [ item for item in subset ]

# a = []
# for i in subset :
#    a.append(i)
