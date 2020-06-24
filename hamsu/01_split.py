import numpy as np

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
    print(type(aaa))                                # i가 0부터 5까지 들어가므로 6번 실행된다는 말, 6번 실행 후에는 for문 나와바리를 벗어나서 
                                                    # aaa 의 type 인 list가 출력
    
    return np.array(aaa)                            # 최종적인 반환 값은 aaa            
